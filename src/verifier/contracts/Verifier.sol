// Modified from https://github.com/appliedzkp/semaphore/blob/master/contracts/sol/verifier.sol
// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;
pragma experimental ABIEncoderV2;
import "./Pairing.sol";
// import "./PolyCoeff.sol";
import { Constants } from "./Constants.sol";
// import "@openzeppelin/contracts/utils/Strings.sol";

contract Verifier is Constants {

    using Pairing for *;

    // PolyCoeff polyCoeff;

    event verifyResult(bool result);
    // description in F below
    uint256[21] Proof = [
        uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
        uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225),
        uint256(20900429899291009073299289469660149716785596251491300692035681492016939179257),
        uint256(17015433841169487450406840425175642150965883946544965612916879717959251917877),
        uint256(433691023568696153828599652727177493671905883454953868604074871528381220097),
        uint256(10320531664273889342578906132837836076031009559738199149165995480549188738219),
        uint256(5841514956038981489672186264970094693930045609713312309929445225955171101947),
        uint256(5021024840007099401420413415719005463562614305701472747429400657816854795888),
        uint256(20774065911754029615648528083815650325450840922528127398846513310899898519818),
        uint256(15033194806153309440324576121312701377491848712637707126378339071163915088607),
        uint256(13797601081368067296147105517336555699959400728196376360418452678919689273357),
        uint256(347536244405038501480645670232637364896709701574004323632284243300674648233),
        uint256(14150985670258525975582637074878024272235049272474457475329855381177734047698),
        uint256(2655786006985334435742353147654701867454007390061009682990360208958229883457),
        uint256(19749978915220999581635876327451130211831443864540429941950424972239040718725),
        uint256(18652734379260577017619098828238354228621932558467343220803598526754221068969),
        uint256(19231643171984527412379363884712632725022106910875011856721870145973184299950),
        uint256(11890584382703984881384687349129438589403848436060509080598302278432395889958),
        uint256(19848703903105846067635106277601755727734486581037082434405934553883502968806),
        uint256(7073067203970200196273245091229017314841577127190841947842173598530677375580),
        uint256(11142795172845103846997117758219330284910812886430955732663385421662518242916)
    ];

    uint256[5] Randoms = [
        uint256(21356640755055926883299664242251323519715676831624930462071588778907420237277),
        uint256(21284924740537517593391635090683107806948436131904811688892120057033464016678),
        uint256(21245863740537517593391635090683107806948436131465498421988152500056051919194), // z
        uint256(21310127601857075097195740179019725106470167450176016701474654076975565675629), // alpha
        uint256(21300179784197405894513072539865320928097398620497813074953097024203243212233) // beta
    ];
    // length of longest u,v,w, i.e. longest length of a,b,c, linear
    uint256 N = 2994;

    uint256 evalK = uint256(11598511819595573397693757683043215863237090817957830497519701049476846220233);
    uint256 evalS = Proof[20];

    // SOCC_hscS = [G1Point S_j, uint256 s_j, G1Point W_j]
    uint256[5] SOCC_hscS = [
        uint256(4856595452220518861283513887787149560249872199754148373623768058895650888886),
        uint256(18903793806709064235976701805982270837399009185115750427816521247566249210556),
        uint256(11142795172845103846997117758219330284910812886430955732663385421662518242916),
        uint256(21371202341828297453905088304285070177929072675831707150146249345026206448204),
        uint256(2804295765584777874267821186909392907007222244895837511075392229436674961086)
    ];
    // SOCC_hscW = [uint256 s'_j,G1Point W'_j, G1Point Q_j]
    uint256[5] SOCC_hscW = [
        uint256(10516069979663785554384039720791935537971662430097624533692998141893610132813),
        uint256(4047408704098476939730481302397316844045052246660100325786631118072453155095),
        uint256(3536770324764440576997419732517461268782698510962602892817204324264966117745),
        uint256(14245367996788138561788347307320292816184394224142842975302936827932398609764),
        uint256(4809015920432886449944012129537258787815219963382067822305826740230093200527)
    ];

    Pairing.G1Point hscQv = Pairing.G1Point(
        uint256(2990558601454446836641627272921201963123578559834602407524792456795005493261),
        uint256(13202662347545347168000079135411531495309137838398006144762638337293564502641)
    );
    Pairing.G1Point hscC = Pairing.G1Point(
        uint256(20354021681082976488982424663336191185782081465046767166624106617637273115698),
        uint256(7434894914590215425794637416957100516507193983165584369402225555559623348194)
    );

    uint256 u = uint256(8197071660809916300712359474937866146139983659726903111927633346212682272753);
    uint256 v = uint256(18304223670379022123371455395869338667757056554451611927086438308637909615067);

    // from terminal
    uint256 s_uv = uint256(7791702837224372263152657165191932519240106980588006270334507770946892728366);

    // from Go
    bytes32 message = ethMessageHash("20900429899291009073299289469660149716785596251491300692035681492016939179257, 433691023568696153828599652727177493671905883454953868604074871528381220097");
    bytes sig = hex"19ec5dc5aa05a220cd210a113352596ebf80d06a6f776b8e0c656e50a5c5567f1e8a7f23fb27f77ea5b5d42f0e2384facdebebd85f026e2a73e94d4690a40a6801";
    address addr = 0xE448992FdEaF94784bBD8f432d781C061D907985;

    // The G2 generator
    // IMPORTANT: why it's here? because the G2Gen() doesn't work in verify() for unknown reason
    Pairing.G2Point g2Generator = Pairing.G2Point({
        X: [ Constants.SRS_G2_X_0[0], Constants.SRS_G2_X_1[0] ],
        Y: [ Constants.SRS_G2_Y_0[0], Constants.SRS_G2_Y_1[0] ]
    });
    // g
    Pairing.G1Point g = Pairing.G1Point(1, 2);
    // xCommit in verify(), I think it's h in SRS
    Pairing.G2Point SRS_G2_1 = Pairing.G2Point({
        X: [ Constants.SRS_G2_X_0[1], Constants.SRS_G2_X_1[1] ],
        Y: [ Constants.SRS_G2_Y_0[1], Constants.SRS_G2_Y_1[1] ]
    });
    // h^α
    Pairing.G2Point SRS_G2_2 = Pairing.G2Point({
        X: [ SRS_G2_X_0_Pos[2], SRS_G2_X_1_Pos[2] ],
        Y: [ SRS_G2_Y_0_Pos[2], SRS_G2_Y_1_Pos[2] ]
    });
    uint256 yz = mulmod(Randoms[0], Randoms[1], Pairing.BABYJUB_P);
    // z, z < BABYJUB_P
    uint256 z = uint256(15296790970327023790902573209632219845262752031560971643217337210279580213975);
    // S, each element < BABYJUB_P
    uint256[2] S1 = [Randoms[1], Randoms[1]];
    uint256[2] S2 = [yz,  yz];
    uint256[1] S3 = [Randoms[1]];
    uint256[3] S4 = [Randoms[1], Randoms[1], Randoms[1]];
    uint256[1] S5 = [Randoms[1]];
    uint256[1] S6 = [Randoms[1]];
    uint256[1] S7 = [Randoms[1]];
    uint256[1] S8 = [Randoms[1]];
    // gamma [0] + [1]*z + [2]*z^2 + ...
    uint256[2] gamma1 = [Randoms[1], Randoms[1]];
    uint256[2] gamma2 = [yz,  yz];
    uint256[1] gamma3 = [Randoms[1]];
    uint256[3] gamma4 = [Randoms[1], Randoms[1], Randoms[1]];
    uint256[1] gamma5 = [Randoms[1]];
    uint256[1] gamma6 = [Randoms[1]];
    uint256[1] gamma7 = [Randoms[1]];
    uint256[1] gamma8 = [Randoms[1]];


    // Sonic version - Verify a single-point evaluation of a polynominal
    // function verify(
    //     Pairing.G1Point memory _commitment, // F
    //     Pairing.G1Point memory _proof, // W
    //     uint256 _index,  // z
    //     uint256 _value,  // F(z) or v
    //     uint proofIndex
    //     //bool isT
    // ) public view returns (bool) {
    //     // Make sure each parameter is less than the prime q
    //     require(_commitment.X < BABYJUB_P, "Verifier.verifyKZG: _commitment.X is out of range");
    //     require(_commitment.Y < BABYJUB_P, "Verifier.verifyKZG: _commitment.Y is out of range");
    //     require(_proof.X < BABYJUB_P, "Verifier.verifyKZG: _proof.X is out of range");
    //     require(_proof.Y < BABYJUB_P, "Verifier.verifyKZG: _proof.Y is out of range");
    //     require(_index < BABYJUB_P, "Verifier.verifyKZG: _index is out of range");
    //     require(_value < BABYJUB_P, "Verifier.verifyKZG: _value is out of range");
       
    //     Pairing.G1Point memory negProof = Pairing.negate(Pairing.mulScalar(_proof, _index));
    //     Pairing.G1Point memory mulProof = Pairing.plus(Pairing.mulScalar(Constants.G1Gen(), _value), negProof);
    //     Pairing.G1Point memory negCm = Pairing.negate(_commitment);

    //     return Pairing.pairing(_proof, Constants.SRS_G2_hAlphaX1(proofIndex),
    //                             mulProof, Constants.SRS_G2_hAlphaX0(proofIndex));
    //                             //negCm, Constants.SRS_G2_hAlphaXdMax(proofIndex, isT)
    // }

    // KZG version - Verify a single-point evaluation of a polynominal
    /*
     * Verifies a single-point evaluation of a polynominal using the KZG
     * commitment scheme.
     *    - p(X) is a polynominal
     *    - _value = p(_index) 
     *    - commitment = commit(p)
     *    - proof = genProof(p, _index, _value)
     * Returns true if and only if the following holds, and returns false
     * otherwise:
     *     e(commitment - commit([_value]), G2.g) == e(proof, commit([0, 1]) - zCommit)
     * @param _commitment The KZG polynominal commitment.
     * @param _proof The proof.
     * @param _index The x-value at which to evaluate the polynominal.
     * @param _value The result of the polynominal evaluation.
     */
    function verify(
        Pairing.G1Point memory _commitment, // F
        Pairing.G1Point memory _proof, // π
        uint256 _index,  // z
        uint256 _value  // F(z) or v
        //uint proofIndex
    ) public view returns (bool) {
        // Make sure each parameter is less than the prime q
        require(_commitment.X < BABYJUB_P, "Verifier.verifyKZG: _commitment.X is out of range");
        require(_commitment.Y < BABYJUB_P, "Verifier.verifyKZG: _commitment.Y is out of range");
        require(_proof.X < BABYJUB_P, "Verifier.verifyKZG: _proof.X is out of range");
        require(_proof.Y < BABYJUB_P, "Verifier.verifyKZG: _proof.Y is out of range");
        require(_index < BABYJUB_P, "Verifier.verifyKZG: _index is out of range");
        require(_value < BABYJUB_P, "Verifier.verifyKZG: _value is out of range");
        // Check that 
        //     e(commitment - aCommit, G2.g) == e(proof, xCommit - zCommit)
        //     e(commitment - aCommit, G2.g) / e(proof, xCommit - zCommit) == 1
        //     e(commitment - aCommit, G2.g) * e(proof, xCommit - zCommit) ^ -1 == 1
        //     e(commitment - aCommit, G2.g) * e(-proof, xCommit - zCommit) == 1
        // where:
        //     aCommit = commit([_value]) = SRS_G1_0 * _value
        //     xCommit = commit([0, 1]) = SRS_G2_1
        //     zCommit = commit([_index]) = SRS_G2_1 * _index

        // To avoid having to perform an expensive operation in G2 to compute
        // xCommit - zCommit, we instead check the equivalent equation:
        //     e(commitment - aCommit, G2.g) * e(-proof, xCommit) * e(-proof, -zCommit) == 1
        //     e(commitment - aCommit, G2.g) * e(-proof, xCommit) * e(proof, zCommit) == 1
        //     e(commitment - aCommit, G2.g) * e(-proof, xCommit) * e(index * proof, G2.g) == 1
        //     e((index * proof) + (commitment - aCommit), G2.g) * e(-proof, xCommit) == 1

        // Compute commitment - aCommitment
        Pairing.G1Point memory commitmentMinusA = Pairing.plus(
            _commitment,
            Pairing.negate(
                Pairing.mulScalar(Constants.G1Gen(), _value)
            )
        );

        // Negate the proof
        Pairing.G1Point memory negProof = Pairing.negate(_proof);

        // Compute index * proof
        Pairing.G1Point memory indexMulProof = Pairing.mulScalar(_proof, _index);

        // Return true if and only if
        // e((index * proof) + (commitment - aCommitment), G2.g) * e(-proof, xCommit) == 1
        return Pairing.pairing(
            Pairing.plus(indexMulProof, commitmentMinusA),
            g2Generator, // 
            negProof,
            SRS_G2_1
        );
    }

    // midified sonic verifier
    function verifySonic(
        // uint256[21] memory Proof,
        // uint256[2] memory Randoms
    ) public returns (bool) {

        // simulate calculating kY
        // uint256 ky = evalKPoly();
        // // // simulate calculating sXY
        // uint256 sx = evalXPoly();
        // uint256 sy = evalXPoly();

        //uint256 yz = mulmod(Randoms[0], Randoms[1], Pairing.BABYJUB_P);

        // y^N for halo implementation style
        // uint256 y_n = expMod(Randoms[1], N, BABYJUB_P);
        // t for halo implementation style
        // uint256 t = addmod(mulmod(addmod(Proof[6], Proof[9], Pairing.BABYJUB_P), 
        //                           addmod(addmod(Proof[12], 
        //                                         Proof[15], Pairing.BABYJUB_P), 
        //                                  evalS, Pairing.BABYJUB_P), Pairing.BABYJUB_P),
        //                     mulmod((BABYJUB_P - evalK), y_n, BABYJUB_P),
        //                     BABYJUB_P);

        uint256 t = addmod(mulmod(addmod(Proof[6], Proof[9], Pairing.BABYJUB_P), 
                                  addmod(addmod(Proof[12], 
                                                Proof[15], Pairing.BABYJUB_P), 
                                         evalS, Pairing.BABYJUB_P), Pairing.BABYJUB_P),
                            (BABYJUB_P - evalK), BABYJUB_P);


        bool verifySignature = recover(message, sig) == addr;
        // bool result = verify(Pairing.G1Point(Proof[0], Proof[1]), // aLocal
        //               Pairing.G1Point(Proof[7], Proof[8]),
        //               Randoms[1], 
        //               Proof[6]) &&
        //         verify(Pairing.G1Point(Proof[0], Proof[1]), // bLocal
        //               Pairing.G1Point(Proof[13], Proof[14]),
        //               yz,
        //               Proof[12]) &&
        //         verify(Pairing.G1Point(Proof[2], Proof[3]), // aRaw
        //               Pairing.G1Point(Proof[10], Proof[11]),
        //               Randoms[1],
        //               Proof[9]) &&
        //         verify(Pairing.G1Point(Proof[2], Proof[3]), // bRaw
        //               Pairing.G1Point(Proof[16], Proof[17]),
        //               yz,
        //               Proof[15]) &&
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                               
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                            // pcV srsLocal (srsD srsLocal) commitK y (k, wk)
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                            // pcV srsLocal (srsD srsLocal) commitC y (c, wc)
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                            // pcV srsLocal (srsD srsLocal) commitC yOld (cOld, wcOld)
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                            //, pcV srsLocal (srsD srsLocal) commitC yNew (cNew, wcNew)
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                            //, pcV srsLocal (srsD srsLocal) commitS z (s, ws)
        //         verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                            //, pcV srsLocal (srsD srsLocal) commitSOld z (sOld, wsOld)
        //          verify(Pairing.G1Point(Proof[4], Proof[5]), // t
        //               Pairing.G1Point(Proof[18], Proof[19]),
        //               Randoms[1],
        //               t) &&                                           //, pcV srsLocal (srsD srsLocal) commitSNew z (sNew, wsNew)
        //         verifySignature &&
        //         Proof[6] == Proof[7] && // c_old == s_old
        //         Proof[8] == Proof[9] && // c_new == s_new
        //         Proof[10] == Proof[11]; // c == s

        // temporary code for estimating gas cost, the above is correct version
        bool result = verify(Pairing.G1Point(Proof[0], Proof[1]), // aLocal
                      Pairing.G1Point(Proof[7], Proof[8]),
                      Randoms[1], 
                      Proof[6]);
        result = verify(Pairing.G1Point(Proof[0], Proof[1]), // bLocal
                      Pairing.G1Point(Proof[13], Proof[14]),
                      yz,
                      Proof[12]);
        result = verify(Pairing.G1Point(Proof[2], Proof[3]), // aRaw
                      Pairing.G1Point(Proof[10], Proof[11]),
                      Randoms[1],
                      Proof[9]);
        result = verify(Pairing.G1Point(Proof[2], Proof[3]), // bRaw
                      Pairing.G1Point(Proof[16], Proof[17]),
                      yz,
                      Proof[15]);
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                              
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                            // pcV srsLocal (srsD srsLocal) commitK y (k, wk)
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                            // pcV srsLocal (srsD srsLocal) commitC y (c, wc)
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                            // pcV srsLocal (srsD srsLocal) commitC yOld (cOld, wcOld)
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                            //, pcV srsLocal (srsD srsLocal) commitC yNew (cNew, wcNew)
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                            //, pcV srsLocal (srsD srsLocal) commitS z (s, ws)
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                            //, pcV srsLocal (srsD srsLocal) commitSOld z (sOld, wsOld)
        result = verify(Pairing.G1Point(Proof[4], Proof[5]), // t
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);                                           //, pcV srsLocal (srsD srsLocal) commitSNew z (sNew, wsNew)
        result = verifySignature;
        result = Proof[6] == Proof[7]; // c_old == s_old
        result = Proof[8] == Proof[9]; // c_new == s_new
        result = Proof[10] == Proof[11]; // c == s
        emit verifyResult(result);
        return result;
    }

    // improvement for batched commitments
    /*
    we check e<g^w[α],h^α> e<g^w[α]g^-z,h> == RHS
     */
    function verifySonicBatched(
        // uint256[21] memory Proof,
        // uint256[2] memory Randoms
    ) public returns (bool) {

        // simulate calculating kY
        // uint256 ky = evalKPoly();
        // // // simulate calculating sXY
        // uint256 sx = evalXPoly();
        // uint256 sy = evalXPoly();

        
        // y^N for halo implementation style
        // uint256 y_n = expMod(Randoms[1], N, BABYJUB_P);

        // t for halo implementation style
        // uint256 t = addmod(mulmod(addmod(Proof[6], Proof[9], Pairing.BABYJUB_P), 
        //                           addmod(addmod(Proof[12], 
        //                                         Proof[15], Pairing.BABYJUB_P), 
        //                                  evalS, Pairing.BABYJUB_P), Pairing.BABYJUB_P),
        //                     mulmod((BABYJUB_P - evalK), y_n, BABYJUB_P),
        //                     BABYJUB_P);

        // uint256 t = addmod(mulmod(addmod(Proof[6], Proof[9], Pairing.BABYJUB_P), 
        //                           addmod(addmod(Proof[12], 
        //                                         Proof[15], Pairing.BABYJUB_P), 
        //                                  evalS, Pairing.BABYJUB_P), Pairing.BABYJUB_P),
        //                     (BABYJUB_P - evalK), BABYJUB_P);


        bool verifySignature = recover(message, sig) == addr;

        // F
        Pairing.G1Point[8] memory F = [
                                    Pairing.G1Point(Proof[0], Proof[1]),//R0
                                    Pairing.G1Point(Proof[2], Proof[3]), // Rj
                                    Pairing.G1Point(Proof[4], Proof[5]), //T
                                    Pairing.G1Point(Proof[6], Proof[7]), //C
                                    Pairing.G1Point(Proof[8], Proof[9]), //Ck
                                    Pairing.G1Point(Proof[10], Proof[11]), //S
                                    Pairing.G1Point(Proof[12], Proof[13]), //Sold
                                    Pairing.G1Point(Proof[14], Proof[15])  //Snew
                                    ];
        // Z(T / Si)[z]
        uint256 Z1 = z_calculation(1); // i = 1
        uint256 Z2 = z_calculation(2);
        uint256 Z3 = z_calculation(3);
        uint256 Z4 = z_calculation(4);
        uint256 Z5 = z_calculation(5);
        uint256 Z6 = z_calculation(6);
        uint256 Z7 = z_calculation(7);
        uint256 Z8 = z_calculation(8);

        // the prove, g^p[α], g^w[α]
        Pairing.G1Point[2] memory pi = [
            G1Gen(),
            G1Gen()
        ];
        // product_result is the first G1 element of the RHS pairing
        // first calculate the uppercase Pi product
        Pairing.G1Point memory product_result = Pairing.mulScalar(Pairing.plus(F[0], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma1[0] + gamma1[1] * Randoms[2]))), Z1);
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[1], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma2[0] + gamma2[1] * Randoms[2]))), Z2));
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[2], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma3[0]))), Z3));
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[3], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma4[0] + gamma4[1] * Randoms[2] + gamma4[2] * Randoms[2]^2))), Z4));
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[4], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma5[0]))), Z5));
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[5], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma6[0]))), Z6));
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[6], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma7[0]))), Z7));
        product_result = Pairing.plus(product_result, 
                                    Pairing.mulScalar(Pairing.plus(F[7], 
                                                                Pairing.negate(Pairing.mulScalar(g, 
                                                                                gamma8[0]))), Z8));
        // then add the first item before the uppercase Pi product
        product_result = Pairing.plus(product_result,
                                    Pairing.negate(Pairing.plus(pi[0],
                                                    Pairing.mulScalar(g,
                                                                    z_calculation(9)))));//zT[z]

        // check e<g^w[α],h^α>e<g^w[α]g^-z,h> == RHS and other checks
        // bool result = Pairing.pairing_3point(
        //     pi[1],
        //     SRS_G2_2, // h^α, see above
        //     Pairing.plus(pi[1], Pairing.negate(Pairing.mulScalar(g, z))),
        //     SRS_G2_1, // h, see above
        //     Pairing.negate(product_result),
        //     SRS_G2_1
        //     )
        //     && verifySignature
        //     && Proof[6] == Proof[7] // c_old == s_old
        //     && Proof[8] == Proof[9] // c_new == s_new
        //     && Proof[10] == Proof[11]; // c == s

        // temporary code for estimating gas cost, the above is correct version
        // check e<g^w[α],h^α>e<g^w[α]g^-z,h> == RHS and other checks
        bool result = Pairing.pairing_3point(
            pi[1],
            SRS_G2_2, // h^α, see above
            Pairing.plus(pi[1], Pairing.negate(Pairing.mulScalar(g, z))),
            SRS_G2_1, // h, see above
            Pairing.negate(product_result),
            SRS_G2_1
            );
        result = verifySignature;
        result = Proof[6] == Proof[7]; // c_old == s_old
        result = Proof[8] == Proof[9]; // c_new == s_new
        result = Proof[10] == Proof[11]; // c == s
        emit verifyResult(result);
        return result;
    }
    

    function z_calculation (uint256 i)
                            internal view returns (uint256){
        
        uint256 result = 1;
        if (i != 1){
            result = mulmod(mulmod(result, addmod(z, BABYJUB_P - S1[1], BABYJUB_P), BABYJUB_P)
                            , addmod(z, BABYJUB_P - S1[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 2){
            result = mulmod(mulmod(result, addmod(z, BABYJUB_P - S2[1], BABYJUB_P), BABYJUB_P)
                            , addmod(z, BABYJUB_P - S2[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 3){
            result = mulmod(result, addmod(z, BABYJUB_P - S3[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 4){
            result = mulmod(mulmod(mulmod(result, addmod(z, BABYJUB_P - S4[2], BABYJUB_P), BABYJUB_P)
                            , addmod(z, BABYJUB_P - S4[1], BABYJUB_P), BABYJUB_P)
                            , addmod(z, BABYJUB_P - S4[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 5){
            result = mulmod(result, addmod(z, BABYJUB_P - S5[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 6){
            result = mulmod(result, addmod(z, BABYJUB_P - S6[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 7){
            result = mulmod(result, addmod(z, BABYJUB_P - S7[0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 8){
            result = mulmod(result, addmod(z, BABYJUB_P - S8[0], BABYJUB_P), BABYJUB_P);
        }
        return result;
    }

    // SCC
    // function verifySOCC() public returns (bool) {

    //     bool verified = verify(Pairing.G1Point(SOCC_hscS[0], SOCC_hscS[1]), // pcV(bp,srs,S_j,d,z_j,(s_j,W_j)
    //                   Pairing.G1Point(SOCC_hscS[3], SOCC_hscS[4]),
    //                   Randoms[1], 
    //                   SOCC_hscS[2],
    //                   0, true) &&
    //                   verify(Pairing.G1Point(SOCC_hscS[0], SOCC_hscS[1]), // pcV(bp,srs,S_j,d,u,(s'_j,W'_j))
    //                   Pairing.G1Point(SOCC_hscW[1], SOCC_hscW[2]),
    //                   u, 
    //                   SOCC_hscW[0],
    //                   0, true) &&
    //                   verify(hscC,                                      // pcV(bp,srs,C,d,y_j,(s'_j,Q_j)
    //                   Pairing.G1Point(SOCC_hscW[3], SOCC_hscW[4]),
    //                   Randoms[0], 
    //                   SOCC_hscW[0],
    //                   0, true) &&
    //                   verify(hscC,                                      // pcV(bp,srs,C,d,v,(s_uv,Q_v))
    //                   hscQv,
    //                   v, 
    //                   s_uv,
    //                   0, true);
    //     emit verifyResult(verified);
    //     return verified;
    // }

    /**
     * @dev Recover signer address from a message by using their signature
     * @param hash bytes32 message, the hash is the signed message. What is recovered is the signer address.
     * @param signature bytes signature, the signature is generated using web3.eth.sign()
     */
    function recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        bytes32 r;
        bytes32 s;
        uint8 _v;

        // Check the signature length
        if (signature.length != 65) {
            return (address(0));
        }

        // Divide the signature in r, s and v variables
        // ecrecover takes the signature parameters, and the only way to get them
        // currently is to use assembly.
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            r := mload(add(signature, 32))
            s := mload(add(signature, 64))
            _v := byte(0, mload(add(signature, 96)))
        }

        // Version of signature should be 27 or 28, but 0 and 1 are also possible versions
        if (_v < 27) {
            _v += 27;
        }

        // If the version is correct return the signer address
        if (_v != 27 && _v != 28) {
            return (address(0));
        } else {
            // solium-disable-next-line arg-overflow
            return ecrecover(hash, _v, r, s);
        }
    }

    /**
    * @dev prefix a bytes32 value with "\x19Ethereum Signed Message:" and hash the result
    */
    function ethMessageHash(string memory rawCommitment) internal pure returns (bytes32) {
        return keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", keccak256(abi.encodePacked(rawCommitment)))
        );
    }


    //     /*
    //  * @return The polynominal evaluation of a polynominal with the specified
    //  *         coefficients at the given index.
    //  */
     function evalPoly() public view returns (uint256) {

        uint baseOrder = 204;
        uint length = 64;
        uint256 _index = Randoms[0];
        uint256 m = Constants.BABYJUB_P;
        uint256 result = 0;
        uint256 powerOfX = 1;

        for (uint256 i = 0; i < length; i ++) {
            assembly {
                result:= addmod(result, mulmod(powerOfX, i, m), m)
                powerOfX := mulmod(powerOfX, i, m)
            }
        }
        uint256 basePower = invMod(expMod(_index, baseOrder, m), m);
        result = mulmod(basePower, result, m);

        return result;
    }

    /// @dev Modular euclidean inverse of a number (mod p).
    /// @param _x The number
    /// @param _pp The modulus
    /// @return q such that x*q = 1 (mod _pp)
    function invMod(uint256 _x, uint256 _pp) internal pure returns (uint256) {
        require(_x != 0 && _x != _pp && _pp != 0, "Invalid number");
        uint256 q = 0;
        uint256 newT = 1;
        uint256 r = _pp;
        uint256 t;
        while (_x != 0) {
        t = r / _x;
        (q, newT) = (newT, addmod(q, (_pp - mulmod(t, newT, _pp)), _pp));
        (r, _x) = (_x, r - t * _x);
        }

        return q;
    }

    /// @dev Modular exponentiation, b^e % _pp.
    /// Source: https://github.com/androlo/standard-contracts/blob/master/contracts/src/crypto/ECCMath.sol
    /// @param _base base
    /// @param _exp exponent
    /// @param _pp modulus
    /// @return r such that r = b**e (mod _pp)
    function expMod(uint256 _base, uint256 _exp, uint256 _pp) internal pure returns (uint256) {
        require(_pp!=0, "Modulus is zero");

        if (_base == 0)
        return 0;
        if (_exp == 0)
        return 1;

        uint256 r = 1;
        uint256 bit = 57896044618658097711785492504343953926634992332820282019728792003956564819968; // 2 ^ 255
        assembly {
        for { } gt(bit, 0) { }{
            // a touch of loop unrolling for 20% efficiency gain
            r := mulmod(mulmod(r, r, _pp), exp(_base, iszero(iszero(and(_exp, bit)))), _pp)
            r := mulmod(mulmod(r, r, _pp), exp(_base, iszero(iszero(and(_exp, div(bit, 2))))), _pp)
            r := mulmod(mulmod(r, r, _pp), exp(_base, iszero(iszero(and(_exp, div(bit, 4))))), _pp)
            r := mulmod(mulmod(r, r, _pp), exp(_base, iszero(iszero(and(_exp, div(bit, 8))))), _pp)
            bit := div(bit, 16)
        }
        }

        return r;
    }
    

}