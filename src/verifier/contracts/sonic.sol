pragma solidity >=0.4.22 <0.9.0;

contract Pairing {

    uint256 constant PRIME_Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    // uint256 constant BABYJUB_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    struct G1Point {
        uint256 X;
        uint256 Y;
    }

    // Encoding of field elements is: X[0] * z + X[1]
    struct G2Point {
        uint256[2] X;
        uint256[2] Y;
    }

    /*
     * @return The negation of p, i.e. p.plus(p.negate()) should be zero. 
     */
    function negate(G1Point memory p) internal pure returns (G1Point memory) {

        // The prime q in the base field F_q for G1
        if (p.X == 0 && p.Y == 0) {
            return G1Point(0, 0);
        } else {
            return G1Point(p.X, PRIME_Q - (p.Y % PRIME_Q));
        }
    }

    /*
     * @return The sum of two points of G1
     */
    function plus(
        G1Point memory p1,
        G1Point memory p2
    ) internal view returns (G1Point memory r) {

        uint256[4] memory input;
        input[0] = p1.X;
        input[1] = p1.Y;
        input[2] = p2.X;
        input[3] = p2.Y;
        bool success;

        // solium-disable-next-line security/no-inline-assembly
        assembly {
            success := staticcall(sub(gas(), 2000), 6, input, 0xc0, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }

        require(success, "pairing-add-failed");
    }

    /*
     * @return The point multiplication of p and q. 
     */
    // function MulPoint(G1Point memory p, G1Point memory q) internal pure returns (G1Point memory) {

    //     // The prime q in the base field F_q for G1
    //     return G1Point(mulmod(p.X, q.X, PRIME_Q), mulmod(p.Y, q.Y, PRIME_Q));
        
    // }

    /*
     * @return The product of a point on G1 and a scalar, i.e.
     *         p == p.scalar_mul(1) and p.plus(p) == p.scalar_mul(2) for all
     *         points p.
     */
    function mulScalar(G1Point memory p, uint256 s) internal view returns (G1Point memory r) {

        uint256[3] memory input;
        input[0] = p.X;
        input[1] = p.Y;
        input[2] = s;
        bool success;
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            success := staticcall(sub(gas(), 2000), 7, input, 0x80, r, 0x60)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require (success, "pairing-mul-failed");
    }

    /* @return The result of computing the pairing check
     *         e(p1[0], p2[0]) *  .... * e(p1[n], p2[n]) == 1
     *         For example,
     *         pairing([P1(), P1().negate()], [P2(), P2()]) should return true.
     */
    // function pairing(
    //     G1Point memory a1,
    //     G2Point memory a2,
    //     G1Point memory b1,
    //     G2Point memory b2
    // ) internal view returns (bool) {

    //     G1Point[2] memory p1 = [a1, b1];
    //     G2Point[2] memory p2 = [a2, b2];

    //     uint256 inputSize = 12;
    //     uint256[] memory input = new uint256[](inputSize);

    //     for (uint256 i = 0; i < 2; i++) {
    //         uint256 j = i * 6;
    //         input[j + 0] = p1[i].X;
    //         input[j + 1] = p1[i].Y;
    //         input[j + 2] = p2[i].X[0];
    //         input[j + 3] = p2[i].X[1];
    //         input[j + 4] = p2[i].Y[0];
    //         input[j + 5] = p2[i].Y[1];
    //     }

    //     uint256[1] memory out;
    //     bool success;
    //     // uint256 len = inputSize * 0x20;
    //     // solium-disable-next-line security/no-inline-assembly
    //     assembly {
    //         success := staticcall(sub(gas(), 2000), 8, add(input, 0x20), mul(inputSize, 0x20), out, 0x20)
    //         // Use "invalid" to make gas estimation work
    //         switch success case 0 { invalid() }
    //     }
    //     require(success, "pairing-opcode-failed");

    //     return out[0] != 0;
    // }

    /* @return The result of computing the 3-point pairing check
     *         e(p1[0], p2[0]) *  .... * e(p1[n], p2[n]) == 1
     *         For example,
     *         pairing([P1(), P1().negate()], [P2(), P2()]) should return true.
     */
    function pairing_3point(
        G1Point memory a1,
        G2Point memory a2,
        G1Point memory b1,
        G2Point memory b2,
        G1Point memory c1,
        G2Point memory c2
    ) internal view returns (bool) {

        G1Point[3] memory p1 = [a1, b1, c1];
        G2Point[3] memory p2 = [a2, b2, c2];

        uint256 inputSize = 18;
        uint256[] memory input = new uint256[](inputSize);

        for (uint256 i = 0; i < 3; i++) {
            uint256 j = i * 6;
            input[j + 0] = p1[i].X;
            input[j + 1] = p1[i].Y;
            input[j + 2] = p2[i].X[0];
            input[j + 3] = p2[i].X[1];
            input[j + 4] = p2[i].Y[0];
            input[j + 5] = p2[i].Y[1];
        }

        uint256[1] memory out;
        bool success;
        //uint256 len = inputSize * 0x20;
        // solium-disable-next-line security/no-inline-assembly
        assembly {
            success := staticcall(sub(gas(), 2000), 0x8, add(input, 0x20), mul(inputSize, 0x20), out, 0x20)
            // Use "invalid" to make gas estimation work
            switch success case 0 { invalid() }
        }
        require(success, "pairing-opcode-failed");

        return out[0] != 0;
        // return true;
    }



    // struct 
    uint256 constant BABYJUB_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    
    // h^αx^(d-max)
    G2Point t_hxdmax = G2Point(
        [11559732032986387107991004021392285783925812861821192530917403151452391805634,
        10857046999023057135944570762232829481370756359578518086990519993285655852781],
        [4082367875863433681332203403145435568316851327593401208105741076214120093531,
        8495653923123431417604973247489272438418190587263600148770280649306958101930]
    );
    // h^αx^(d-max+1)
    // G2Point t_hxdmaxplusone = G2Point(
    //     [11559732032986387107991004021392285783925812861821192530917403151452391805634,
    //     10857046999023057135944570762232829481370756359578518086990519993285655852781],
    //     [4082367875863433681332203403145435568316851327593401208105741076214120093531,
    //     8495653923123431417604973247489272438418190587263600148770280649306958101930]
    // );
    // h
    G2Point SRS_G2_1 = G2Point({
        X: [ uint256(0x21a808dad5c50720fb7294745cf4c87812ce0ea76baa7df4e922615d1388f25a), uint256(0x04c5e74c85a87f008a2feb4b5c8a1e7f9ba9d8eb40eb02e70139c89fb1c505a9) ],
        Y: [ uint256(0x204b66d8e1fadc307c35187a6b813be0b46ba1cd720cd1c4ee5f68d13036b4ba), uint256(0x2d58022915fc6bc90e036e858fbc98055084ac7aff98ccceb0e3fde64bc1a084) ]
    });


    // // h^αx^(d-max)
    // G2Point t_hxdmax = G2Point(
    //     [14502447760486387799059318541209757040844770937862468921929310682431317530875,
    //     2443430939986969712743682923434644543094899517010817087050769422599268135103],
    //     [11721331165636005533649329538372312212753336165656329339895621434122061690013,
    //     4704672529862198727079301732358554332963871698433558481208245291096060730807]
    // );
    // // h^αx^(d-max+1)
    // G2Point t_hxdmaxplusone = G2Point(
    //     [9753532634097748217842057100292196132812913888453919348849716624821506442403,
    //     11268695302917653497930368204518419275811373844270095345017116270039580435706],
    //     [18234420836750160252624653416906661873432457147248720023481962812575477763504,
    //     14776274355700936137709623009318398493759214126639684828506188800357268553876]
    // );
    // // h
    // G2Point SRS_G2_1 = G2Point({
    //     X: [ uint256(10857046999023057135944570762232829481370756359578518086990519993285655852781), 
    //     uint256(11559732032986387107991004021392285783925812861821192530917403151452391805634) ],
    //     Y: [ uint256(8495653923123431417604973247489272438418190587263600148770280649306958101930), 
    //     uint256(4082367875863433681332203403145435568316851327593401208105741076214120093531) ]
    // });



    event verifyResult(bool result);
    // event checkData(uint256 H);

    
    uint256 yz = mulmod(y, z, BABYJUB_P);
    uint256 z = uint256(2);
    uint256 y = uint256(3);
    uint256 beta = uint256(1);

    // number of constraints?
    uint256 N = 232;
    uint256 z_n = expMod(z, N, BABYJUB_P);

    // struct UserCommitments{

    //     // ECDSA signature
    //     bytes32 message;
    //     bytes sig;
    //     address addr;

    //     // d_j when j=1

    //     // S

    //     // gamma(z) = gamma[0] + gamma[1]*z + gamma[2]*z^2 + ...
    //     uint256[1] gamma1;
    //     uint256[1] gamma2;
    //     uint256[2] gamma3;
    //     uint256[1] gamma4;
    //     uint256[1] gamma5;
    //     uint256[2] gamma6;
    //     uint256[1] gamma7;
    //     // the committed prove, g^p[x], g^w[x]
    //     G1Point pi_1;
    //     G1Point pi_2;


    //     // other prover submitted variables
    //     uint256 r_1;
    //     uint256 r_tilde;
    //     uint256 t;
    //     uint256 k;
    //     uint256 s_tilde;
    //     uint256 r_2;
    //     uint256 s_1_tilde;
    //     uint256 s_2_tilde;

    //     // poly commitments, Fs
    //     G1Point D;
    //     G1Point R_tilde;
    //     G1Point R;
    //     G1Point T;
    //     G1Point K;
    //     G1Point S_x;
    //     G1Point S_y;
    // }
    // // too many local variables, so create a struct
    // struct verification_variables{
    //     // Z(T / Si)[z] * β^(i-1)
    //     uint256 Z_beta_1; // i = 1,  etc.
    //     uint256 Z_beta_2;
    //     uint256 Z_beta_3;
    //     uint256 Z_beta_4;

    // }

    // // using batched commitments of sonic version of modified KZG
    // // used for test convenience only
    // function verifySonicBatched(
    // ) public{
        
    //     verifySonicBatchedImpl(UserCommitments(

    //         // ECDSA signature
    //         ethMessageHash("20900429899291009073299289469660149716785596251491300692035681492016939179257, 433691023568696153828599652727177493671905883454953868604074871528381220097"),
    //         hex"19ec5dc5aa05a220cd210a113352596ebf80d06a6f776b8e0c656e50a5c5567f1e8a7f23fb27f77ea5b5d42f0e2384facdebebd85f026e2a73e94d4690a40a6801",
    //         0xE448992FdEaF94784bBD8f432d781C061D907985,

    //         // d_j when j=1


    //         // gamma(z) = gamma[0] + gamma[1]*z + gamma[2]*z^2 + ...
    //         [z],
    //         [z],
    //         [z, yz],
    //         [z],
    //         [z],
    //         [z, 1],
    //         [z],
    //         // the committed prove, g^p[x], g^w[x]
    //         G1Point(8691812572209236787755909190653066973757831274058158466035812001132378232132, 
    //                     17614238606459280067365568753035794956929523929908239924500587786582782167157),
    //         G1Point(2719399906655073640085337759413551928897712879340851091327276976055518598223, 
    //                     20784423063878520578323692594726006668764352581142691563520288637533716186542),


    //         // other prover submitted variables
    //         uint256(4242722527485925673536771641410749919442789995051731087056631951011894193824),
    //         uint256(4235762420456420407298727263734883839454930723612238987609433064826275819168),
    //         uint256(15628478343469073419859464163329062466121770485168599321100051518363659718822),
    //         uint256(8989609048714533841442337639132668757304461060442870621846536246819676694537),
    //         uint256(6768634790311320323561834910035611630507898790565911746538692734203120436633),
    //         uint256(14403089981358372733888048340877043885368791532361516465319358419079853271499),
    //         uint256(20989736645852010730059090887752905113483273019753499524902283284119561041372),
    //         uint256(20989736645852010730059090887752905113483273019753499524902283284119561041372),

    //         // poly commitments, Fs
    //         G1Point(uint256(8021098061900953036260251354319004961720986543079211370887400560040844391796),
    //             uint256(14058471992277462689489968894496750533431399137671466792641892934798592486849)),
    //         G1Point(uint256(11362794564866923065782992234127524046383828716475770775850165393630678513302),
    //             uint256(6351286445590955376401750699710520657105132283096474020050378719502426494648)),
    //         G1Point(uint256(21314345901433505953066250779653689226489001651277903907071883064612879132427),
    //             uint256(5678989354845008225866774570037297655830278689813150286547302989191493822952)),
    //         G1Point(uint256(117856289824492738767550145074241964086085676453381628187452269527947479088),
    //             uint256(8398321730595406620438004680380688229020330346885127310873490199379668721173)),
    //         G1Point(uint256(10935811711329215078573900228397874487452484444342843838558709647446711964975),
    //             uint256(12740820905143376503975568304814073478092535106619449178774030661527709973801)),
    //         G1Point(uint256(18147983606650661809662982452167395327726597071596257186624141873821495240067),
    //             uint256(9051566570125000288475827950874181038226002513175138234491129181216164164259)),
    //         G1Point(uint256(16795591823642793416253914800363052108914641401815054099491991256863227018190),
    //             uint256(18190150497060778624076211852902699629600032853080044790376820709104023547442))
    //     ));
    // }

    // function verifySonicBatchedImpl(
    //     UserCommitments memory cm
    // ) public returns (bool) {

    //     verification_variables memory vars;
    //     // Z(T / Si)[z] * β^(i-1)
    //     vars.Z_beta_1 = mulmod(1, z_calculation(1), BABYJUB_P); // i = 1,  etc.
    //     vars.Z_beta_2 = mulmod(beta, z_calculation(2), BABYJUB_P);
    //     vars.Z_beta_3 = mulmod(mulmod(vars.Z_beta_2, beta, BABYJUB_P), z_calculation(3), BABYJUB_P);
    //     vars.Z_beta_4 = mulmod(mulmod(vars.Z_beta_3, beta, BABYJUB_P), z_calculation(4), BABYJUB_P);
    //     uint256 Z_beta_5 = mulmod(mulmod(vars.Z_beta_4, beta, BABYJUB_P), z_calculation(5), BABYJUB_P);
    //     uint256 Z_beta_6 = mulmod(mulmod(Z_beta_5, beta, BABYJUB_P), z_calculation(6), BABYJUB_P);
    //     uint256 Z_beta_7 = mulmod(mulmod(Z_beta_6, beta, BABYJUB_P), z_calculation(7), BABYJUB_P);
        
    //     // H calculation
    //     G1Point memory H = plus(plus(mulScalar(cm.D, vars.Z_beta_1), mulScalar(cm.R_tilde, vars.Z_beta_2)), mulScalar(cm.R, vars.Z_beta_3));
    //     H = plus(H, mulScalar(cm.T, vars.Z_beta_4));
    //     H = plus(H, mulScalar(cm.K, Z_beta_5));
    //     H = plus(H, mulScalar(cm.S_x, Z_beta_6));
    //     H = plus(H, mulScalar(cm.S_y, Z_beta_7));

    //     // R calculation, denoted  RR because already have a R for one Fcommitment
    //     // first calculate the PI product, and to do this first calculate the power of g after product to reduce gas cost
    //     uint256 power = mulmod(vars.Z_beta_1, BABYJUB_P - cm.gamma1[0], BABYJUB_P);
    //     power = addmod(power, mulmod(vars.Z_beta_2, BABYJUB_P - cm.gamma2[0], BABYJUB_P), BABYJUB_P);
    //     power = addmod(power, mulmod(vars.Z_beta_3, BABYJUB_P - addmod(cm.gamma3[0], mulmod(cm.gamma3[1], z, BABYJUB_P), 
    //                                                             BABYJUB_P), BABYJUB_P), BABYJUB_P);
    //     power = addmod(power, mulmod(vars.Z_beta_4, BABYJUB_P - cm.gamma4[0], BABYJUB_P), BABYJUB_P);
    //     // power = addmod(power, mulmod(Z_beta_4, BABYJUB_P - addmod(addmod(gamma4[0], 
    //     //                                                                         mulmod(gamma4[1], z, BABYJUB_P), 
    //     //                                                                         BABYJUB_P),
    //     //                                                                         mulmod(gamma4[2], 
    //     //                                                                         mulmod(z, z, 
    //     //                                                                         BABYJUB_P), 
    //     //                                                         BABYJUB_P),
    //     //                                                     BABYJUB_P),
    //     //                             BABYJUB_P), BABYJUB_P);
    //     power = addmod(power, mulmod(Z_beta_5, BABYJUB_P - cm.gamma5[0], BABYJUB_P), BABYJUB_P);
    //     power = addmod(power, mulmod(Z_beta_6, BABYJUB_P - addmod(cm.gamma6[0], mulmod(cm.gamma6[1], z, BABYJUB_P), 
    //                                                             BABYJUB_P), BABYJUB_P), BABYJUB_P);
    //     power = addmod(power, mulmod(Z_beta_7, BABYJUB_P - cm.gamma7[0], BABYJUB_P), BABYJUB_P);
    //     // calculate the PI product
    //     G1Point memory RR = mulScalar(G1Point(1, 2), power);

    //     // then add the first item before the uppercase Pi product
    //     //g^p[x]·zT[z]
    //     RR = plus(RR, negate(plus(cm.pi_1,
    //                                                     mulScalar(G1Point(1, 2),
    //                                                                     z_calculation(9)))));
    //     //g^z·w[x]
    //     RR = plus(RR, mulScalar(cm.pi_2, z));
        

    //     H = G1Point(20077374419706392512429686799679330166706149542289304718417735408826647662713, 
    //         14885977252123072602720243771283406369571578426117217829155383966924599215549);
    //     RR = G1Point(5854638644333843355810549205783961330813896657133136761631324628019915576985, 
    //         10388050893327881316839543061910361629748564725587001630616572947848175252412);
    //     // check the equation, then check others
    //     bool result = pairing_3point(
    //         H,
    //         SRS_G2_1,
    //         RR,
    //         t_hxdmax,
    //         negate(cm.pi_2),
    //         t_hxdmaxplusone
    //         );
    //         // && recover(cm.message, cm.sig) == cm.addr //verifySignature
    //         // && cm.r_1 == addmod(cm.r_tilde, mulmod(d, z_n, BABYJUB_P), BABYJUB_P)
    //         // && cm.t == addmod(mulmod(cm.r_1, 
    //         //                       addmod(cm.r_2,
    //         //                              cm.s_tilde, BABYJUB_P), BABYJUB_P),
    //         //                 (BABYJUB_P - cm.k), BABYJUB_P)
    //         // && cm.s_1_tilde == cm.s_2_tilde;

    //     // temporary code for estimating gas cost, the above is correct version
    //     // bool result = pairing_3point(
    //     //     H,
    //     //     SRS_G2_1,
    //     //     RR,
    //     //     t_hxdmax,
    //     //     negate(cm.pi_2),
    //     //     t_hxdmaxplusone
    //     //     );
    //     // result = recover(cm.message, cm.sig) == cm.addr; //verifySignature
    //     // result = cm.r_1 == addmod(cm.r_tilde, mulmod(d, z_n, BABYJUB_P), BABYJUB_P);
    //     // result = cm.t == addmod(mulmod(cm.r_1, 
    //     //                           addmod(cm.r_2,
    //     //                                  cm.s_tilde, BABYJUB_P), BABYJUB_P),
    //     //                     (BABYJUB_P - cm.k), BABYJUB_P);
    //     // result = cm.s_1_tilde == cm.s_2_tilde;

    //     emit verifyResult(result);
    //     // emit checkData(H.X);
        
    //     return result;
    // }
    

    // function z_calculation (uint256 i)
    //                         internal view returns (uint256){
        
    //     uint256 result = 1;
    //     if (i != 1){
    //         result = mulmod(result, addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     if (i != 2){
    //         result = mulmod(result, addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     if (i != 3){
    //         result = mulmod(mulmod(result, addmod(z, BABYJUB_P - [z, yz][1], BABYJUB_P), BABYJUB_P)
    //                         , addmod(z, BABYJUB_P - [z, yz][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     if (i != 4){
    //         result = mulmod(result, addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
    //         // result = mulmod(mulmod(mulmod(result, addmod(z, BABYJUB_P - [z][2], BABYJUB_P), BABYJUB_P)
    //         //                 , addmod(z, BABYJUB_P - [z][1], BABYJUB_P), BABYJUB_P)
    //         //                 , addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     if (i != 5){
    //         result = mulmod(result, addmod(z, BABYJUB_P - [y][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     if (i != 6){
    //         result = mulmod(mulmod(result, addmod(z, BABYJUB_P - [z, 1][1], BABYJUB_P), BABYJUB_P)
    //                         , addmod(z, BABYJUB_P - [z, 1][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     if (i != 7){
    //         result = mulmod(result, addmod(z, BABYJUB_P - [y][0], BABYJUB_P), BABYJUB_P);
    //     }
    //     return result;
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
    //  function evalPoly() public view returns (uint256) {

    //     uint baseOrder = 204;
    //     uint length = 64;
    //     uint256 _index = y;
    //     uint256 m = BABYJUB_P;
    //     uint256 result = 0;
    //     uint256 powerOfX = 1;

    //     for (uint256 i = 0; i < length; i ++) {
    //         assembly {
    //             result:= addmod(result, mulmod(powerOfX, i, m), m)
    //             powerOfX := mulmod(powerOfX, i, m)
    //         }
    //     }
    //     uint256 basePower = invMod(expMod(_index, baseOrder, m), m);
    //     result = mulmod(basePower, result, m);

    //     return result;
    // }

    /// @dev Modular euclidean inverse of a number (mod p).
    /// @param _x The number
    /// @param _pp The modulus
    /// @return q such that x*q = 1 (mod _pp)
    // function invMod(uint256 _x, uint256 _pp) internal pure returns (uint256) {
    //     require(_x != 0 && _x != _pp && _pp != 0, "Invalid number");
    //     uint256 q = 0;
    //     uint256 newT = 1;
    //     uint256 r = _pp;
    //     uint256 t;
    //     while (_x != 0) {
    //     t = r / _x;
    //     (q, newT) = (newT, addmod(q, (_pp - mulmod(t, newT, _pp)), _pp));
    //     (r, _x) = (_x, r - t * _x);
    //     }

    //     return q;
    // }

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











// temporarily this is for original sonic
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

// Original Sonic verison uses:
    // h^α
    G2Point SRS_G2_hAlphaX0 = G2Point(
        [11559732032986387107991004021392285783925812861821192530917403151452391805634,
        10857046999023057135944570762232829481370756359578518086990519993285655852781],
        [4082367875863433681332203403145435568316851327593401208105741076214120093531,
        8495653923123431417604973247489272438418190587263600148770280649306958101930]
    );
    // h^αx
    G2Point SRS_G2_hAlphaX1 = G2Point(
        [11559732032986387107991004021392285783925812861821192530917403151452391805634,
        10857046999023057135944570762232829481370756359578518086990519993285655852781],
        [4082367875863433681332203403145435568316851327593401208105741076214120093531,
        8495653923123431417604973247489272438418190587263600148770280649306958101930]
    );

    // The G1 generator
    function G1Gen() pure internal returns (G1Point memory) {
        return G1Point(1, 2);
    }

    // // Sonic proofs
    // // D_j when j=1
    // G1Point pi_D = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_R_tilde = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_R1 = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_R2 = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_T = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_K = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_S_x1 = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_S_x2 = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point pi_S_y = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // uint256 r_1 = uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986);
    // uint256 r_tilde = uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225);
    // uint256 t = uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986);
    // uint256 k = uint256(11598511819595573397693757683043215863237090817957830497519701049476846220233);
    // uint256 s_tilde = uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986);
    // uint256 r_2 = uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225);
    // uint256 s_1_tilde = uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986);
    // uint256 s_2_tilde = uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225);
    // // poly commitments, Fs
    // G1Point D = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point R_tilde = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point R = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point T = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point K = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point S_x = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // G1Point S_y = G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
    //     uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225));
    // bytes32 message = ethMessageHash("20900429899291009073299289469660149716785596251491300692035681492016939179257, 433691023568696153828599652727177493671905883454953868604074871528381220097");
    // bytes sig = hex"19ec5dc5aa05a220cd210a113352596ebf80d06a6f776b8e0c656e50a5c5567f1e8a7f23fb27f77ea5b5d42f0e2384facdebebd85f026e2a73e94d4690a40a6801";
    // address addr = 0xE448992FdEaF94784bBD8f432d781C061D907985;

    struct UserCommitments{
        G1Point pi_D;
        G1Point pi_R_tilde;
        G1Point pi_R1;
        G1Point pi_R2;
        G1Point pi_T;
        G1Point pi_K;
        G1Point pi_S_x1;
        G1Point pi_S_x2;
        G1Point pi_S_y;
        uint256 r_1;
        uint256 r_tilde;
        uint256 t;
        uint256 k;
        uint256 s_tilde;
        uint256 r_2;
        uint256 s_1_tilde;
        uint256 s_2_tilde;
        // poly commitments; Fs
        G1Point D;
        G1Point R_tilde;
        G1Point R;
        G1Point T;
        G1Point K;
        G1Point S_x;
        G1Point S_y;
        bytes32 message;
        bytes sig;
        address addr;

        // d_j when j=1
        uint256 d;
    }


// Sonic version - Verify a single-point evaluation of a polynominal
    function verify(
        G1Point memory _commitment, // F
        G1Point memory _proof, // W
        uint256 _index,  // z
        uint256 _value  // F(z) or v
        // uint proofIndex,
        // bool isT
    ) public view returns (bool) {
        // Make sure each parameter is less than the prime q
        require(_commitment.X < BABYJUB_P, "Verifier.verifyKZG: _commitment.X is out of range");
        require(_commitment.Y < BABYJUB_P, "Verifier.verifyKZG: _commitment.Y is out of range");
        require(_proof.X < BABYJUB_P, "Verifier.verifyKZG: _proof.X is out of range");
        require(_proof.Y < BABYJUB_P, "Verifier.verifyKZG: _proof.Y is out of range");
        require(_index < BABYJUB_P, "Verifier.verifyKZG: _index is out of range");
        require(_value < BABYJUB_P, "Verifier.verifyKZG: _value is out of range");
       
        G1Point memory negProof = negate(mulScalar(_proof, _index));
        G1Point memory mulProof = plus(mulScalar(G1Gen(), _value), negProof);

        return pairing_3point(_proof, SRS_G2_hAlphaX1,
                                mulProof, SRS_G2_hAlphaX0,
                                _commitment, t_hxdmax);
    }

    // using sonic commitments of sonic version of modified KZG
    // used for test convenience only
    function verifySonic(
    ) public{
        
        verifySonicImpl(UserCommitments(
            // Sonic proofs
            // D_j when j=1
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
            uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225),
            uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
            uint256(11598511819595573397693757683043215863237090817957830497519701049476846220233),
            uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
            uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225),
            uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
            uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225),
            // poly commitments, Fs
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            G1Point(uint256(20435686948508171234472206488737953800505595616105823290561271581793730135986),
                uint256(7613038940582986439878577004424311309737615170791456916446723479068371769225)),
            ethMessageHash("20900429899291009073299289469660149716785596251491300692035681492016939179257, 433691023568696153828599652727177493671905883454953868604074871528381220097"),
            hex"19ec5dc5aa05a220cd210a113352596ebf80d06a6f776b8e0c656e50a5c5567f1e8a7f23fb27f77ea5b5d42f0e2384facdebebd85f026e2a73e94d4690a40a6801",
            0xE448992FdEaF94784bBD8f432d781C061D907985,

            uint256(5935276955487653023739309505454236985420338951434412891728899533829623384892)
            ));
    }


    // sonic verifier

    function verifySonicImpl(
        // uint256[21] memory Proof,
        // uint256[2] memory Randoms
        UserCommitments memory cm
    ) public returns (bool) {

        // simulate calculating kY
        // uint256 ky = evalKPoly();
        // // // simulate calculating sXY
        // uint256 sx = evalXPoly();
        // uint256 sy = evalXPoly();

        // uint256 yz = mulmod(y, z, BABYJUB_P);

        // y^N for halo implementation style
        // uint256 y_n = expMod(z, N, BABYJUB_P);
        // t for halo implementation style
        // uint256 t = addmod(mulmod(addmod(Proof[6], Proof[9], BABYJUB_P), 
        //                           addmod(addmod(Proof[12], 
        //                                         Proof[15], BABYJUB_P), 
        //                                  evalS, BABYJUB_P), BABYJUB_P),
        //                     mulmod((BABYJUB_P - evalK), y_n, BABYJUB_P),
        //                     BABYJUB_P);



        // bool result = verify(D,
        //               pi_D,
        //               z, 
        //               d);
        //     && verify(R_tilde,
        //               pi_R_tilde,
        //               z,
        //               r_tilde);
        //     && verify(R,
        //               pi_R1,
        //               z,
        //               r_1);
        //     && verify(R,
        //               pi_R2,
        //               yz,
        //               r_2);
        //     && verify(T,
        //               pi_T,
        //               z,
        //               t);
        //     && verify(K,
        //               pi_K,
        //               y,
        //               k);
        //     && verify(S_x,
        //               pi_S_x1,
        //               z,
        //               s_tilde);
        //     && verify(S_x,
        //               pi_S_x2,
        //               1,
        //               s_1_tilde);
        //     && verify(S_y,
        //               pi_S_y,
        //               y,
        //               s_2_tilde);
        //     && recover(message, sig) == addr; //verifySignature
        //     && r_1 == addmod(r_tilde, mulmod(d, z_n, BABYJUB_P), BABYJUB_P);
        //     && t == addmod(mulmod(r_1, 
        //                           addmod(r_2,
        //                                  s_tilde, BABYJUB_P), BABYJUB_P),
        //                     (BABYJUB_P - k), BABYJUB_P);
        //     && s_1_tilde == s_2_tilde;

        // temporary code for estimating gas cost, the above is correct version
        bool result = verify(cm.D,
                      cm.pi_D,
                      z, 
                      cm.d);
        result = verify(cm.R_tilde,
                      cm.pi_R_tilde,
                      z,
                      cm.r_tilde);
        result = verify(cm.R,
                      cm.pi_R1,
                      z,
                      cm.r_1);
        result = verify(cm.R,
                      cm.pi_R2,
                      yz,
                      cm.r_2);
        result = verify(cm.T,
                      cm.pi_T,
                      z,
                      cm.t);
        result = verify(cm.K,
                      cm.pi_K,
                      y,
                      cm.k);
        result = verify(cm.S_x,
                      cm.pi_S_x1,
                      z,
                      cm.s_tilde);
        result = verify(cm.S_x,
                      cm.pi_S_x2,
                      1,
                      cm.s_1_tilde);
        result = verify(cm.S_y,
                      cm.pi_S_y,
                      y,
                      cm.s_2_tilde);
        result = recover(cm.message, cm.sig) == cm.addr; //verifySignature
        result = cm.r_1 == addmod(cm.r_tilde, mulmod(cm.d, z_n, BABYJUB_P), BABYJUB_P);
        result = cm.t == addmod(mulmod(cm.r_1, 
                                  addmod(cm.r_2,
                                         cm.s_tilde, BABYJUB_P), BABYJUB_P),
                            (BABYJUB_P - cm.k), BABYJUB_P);
        result = cm.s_1_tilde == cm.s_2_tilde;

        emit verifyResult(result);
        return result;
    }
}