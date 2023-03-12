// Modified from https://github.com/appliedzkp/semaphore/blob/master/contracts/sol/verifier.sol
// SPDX-License-Identifier: MIT

// please notice that current code may be temporary code for cost estimation, and right version is annotated

pragma solidity >=0.4.22 <0.9.0;
pragma experimental ABIEncoderV2;
import "./Pairing.sol";
// import "./PolyCoeff.sol";
import { Constants } from "./Constants.sol";
// import "@openzeppelin/contracts/utils/Strings.sol";

contract Verifier is Constants {

    using Pairing for *;

    event verifyResult(bool result);
    // event checkData(uint256 H);

    // d_j when j=1
    uint256 d = uint256(5935276955487653023739309505454236985420338951434412891728899533829623384892);
    uint256 yz = mulmod(y, z, BABYJUB_P);
    uint256 z = uint256(2);
    uint256 y = uint256(3);
    uint256 beta = uint256(1);

    // number of constraints?
    uint256 N = 232;
    uint256 z_n = expMod(z, N, BABYJUB_P);

    struct UserCommitments{

        // ECDSA signature
        bytes32 message;
        bytes sig;
        address addr;

        // d_j when j=1

        // S

        // gamma(z) = gamma[0] + gamma[1]*z + gamma[2]*z^2 + ...
        uint256[1] gamma1;
        uint256[1] gamma2;
        uint256[2] gamma3;
        uint256[1] gamma4;
        uint256[1] gamma5;
        uint256[2] gamma6;
        uint256[1] gamma7;
        // the committed prove, g^p[x], g^w[x]
        Pairing.G1Point pi_1;
        Pairing.G1Point pi_2;


        // other prover submitted variables
        uint256 r_1;
        uint256 r_tilde;
        uint256 t;
        uint256 k;
        uint256 s_tilde;
        uint256 r_2;
        uint256 s_1_tilde;
        uint256 s_2_tilde;

        // poly commitments, Fs
        Pairing.G1Point D;
        Pairing.G1Point R_tilde;
        Pairing.G1Point R;
        Pairing.G1Point T;
        Pairing.G1Point K;
        Pairing.G1Point S_x;
        Pairing.G1Point S_y;
    }
    // too many local variables, so create a struct
    struct verification_variables{
        // Z(T / Si)[z] * β^(i-1)
        uint256 Z_beta_1; // i = 1,  etc.
        uint256 Z_beta_2;
        uint256 Z_beta_3;
        uint256 Z_beta_4;

    }

    // using batched commitments of sonic version of modified KZG
    // used for test convenience only
    function verifySonicBatched(
    ) public{
        
        verifySonicBatchedImpl(UserCommitments(

            // ECDSA signature
            ethMessageHash("20900429899291009073299289469660149716785596251491300692035681492016939179257, 433691023568696153828599652727177493671905883454953868604074871528381220097"),
            hex"19ec5dc5aa05a220cd210a113352596ebf80d06a6f776b8e0c656e50a5c5567f1e8a7f23fb27f77ea5b5d42f0e2384facdebebd85f026e2a73e94d4690a40a6801",
            0xE448992FdEaF94784bBD8f432d781C061D907985,

            // d_j when j=1


            // gamma(z) = gamma[0] + gamma[1]*z + gamma[2]*z^2 + ...
            [z],
            [z],
            [z, yz],
            [z],
            [z],
            [z, 1],
            [z],
            // the committed prove, g^p[x], g^w[x]
            Pairing.G1Point(8691812572209236787755909190653066973757831274058158466035812001132378232132, 
                        17614238606459280067365568753035794956929523929908239924500587786582782167157),
            Pairing.G1Point(2719399906655073640085337759413551928897712879340851091327276976055518598223, 
                        20784423063878520578323692594726006668764352581142691563520288637533716186542),


            // other prover submitted variables
            uint256(4242722527485925673536771641410749919442789995051731087056631951011894193824),
            uint256(4235762420456420407298727263734883839454930723612238987609433064826275819168),
            uint256(15628478343469073419859464163329062466121770485168599321100051518363659718822),
            uint256(8989609048714533841442337639132668757304461060442870621846536246819676694537),
            uint256(6768634790311320323561834910035611630507898790565911746538692734203120436633),
            uint256(14403089981358372733888048340877043885368791532361516465319358419079853271499),
            uint256(20989736645852010730059090887752905113483273019753499524902283284119561041372),
            uint256(20989736645852010730059090887752905113483273019753499524902283284119561041372),

            // poly commitments, Fs
            Pairing.G1Point(uint256(8021098061900953036260251354319004961720986543079211370887400560040844391796),
                uint256(14058471992277462689489968894496750533431399137671466792641892934798592486849)),
            Pairing.G1Point(uint256(11362794564866923065782992234127524046383828716475770775850165393630678513302),
                uint256(6351286445590955376401750699710520657105132283096474020050378719502426494648)),
            Pairing.G1Point(uint256(21314345901433505953066250779653689226489001651277903907071883064612879132427),
                uint256(5678989354845008225866774570037297655830278689813150286547302989191493822952)),
            Pairing.G1Point(uint256(117856289824492738767550145074241964086085676453381628187452269527947479088),
                uint256(8398321730595406620438004680380688229020330346885127310873490199379668721173)),
            Pairing.G1Point(uint256(10935811711329215078573900228397874487452484444342843838558709647446711964975),
                uint256(12740820905143376503975568304814073478092535106619449178774030661527709973801)),
            Pairing.G1Point(uint256(18147983606650661809662982452167395327726597071596257186624141873821495240067),
                uint256(9051566570125000288475827950874181038226002513175138234491129181216164164259)),
            Pairing.G1Point(uint256(16795591823642793416253914800363052108914641401815054099491991256863227018190),
                uint256(18190150497060778624076211852902699629600032853080044790376820709104023547442))
        ));
    }

    function verifySonicBatchedImpl(
        UserCommitments memory cm
    ) public returns (bool) {

        verification_variables memory vars;
        // Z(T / Si)[z] * β^(i-1)
        vars.Z_beta_1 = mulmod(1, z_calculation(1), BABYJUB_P); // i = 1,  etc.
        vars.Z_beta_2 = mulmod(beta, z_calculation(2), BABYJUB_P);
        vars.Z_beta_3 = mulmod(mulmod(vars.Z_beta_2, beta, BABYJUB_P), z_calculation(3), BABYJUB_P);
        vars.Z_beta_4 = mulmod(mulmod(vars.Z_beta_3, beta, BABYJUB_P), z_calculation(4), BABYJUB_P);
        uint256 Z_beta_5 = mulmod(mulmod(vars.Z_beta_4, beta, BABYJUB_P), z_calculation(5), BABYJUB_P);
        uint256 Z_beta_6 = mulmod(mulmod(Z_beta_5, beta, BABYJUB_P), z_calculation(6), BABYJUB_P);
        uint256 Z_beta_7 = mulmod(mulmod(Z_beta_6, beta, BABYJUB_P), z_calculation(7), BABYJUB_P);
        
        // H calculation
        Pairing.G1Point memory H = Pairing.plus(Pairing.plus(Pairing.mulScalar(cm.D, vars.Z_beta_1), Pairing.mulScalar(cm.R_tilde, vars.Z_beta_2)), Pairing.mulScalar(cm.R, vars.Z_beta_3));
        H = Pairing.plus(H, Pairing.mulScalar(cm.T, vars.Z_beta_4));
        H = Pairing.plus(H, Pairing.mulScalar(cm.K, Z_beta_5));
        H = Pairing.plus(H, Pairing.mulScalar(cm.S_x, Z_beta_6));
        H = Pairing.plus(H, Pairing.mulScalar(cm.S_y, Z_beta_7));

        // R calculation, denoted  RR because already have a R for one Fcommitment
        // first calculate the PI product, and to do this first calculate the power of g after product to reduce gas cost
        uint256 power = mulmod(vars.Z_beta_1, BABYJUB_P - cm.gamma1[0], BABYJUB_P);
        power = addmod(power, mulmod(vars.Z_beta_2, BABYJUB_P - cm.gamma2[0], BABYJUB_P), BABYJUB_P);
        power = addmod(power, mulmod(vars.Z_beta_3, BABYJUB_P - addmod(cm.gamma3[0], mulmod(cm.gamma3[1], z, BABYJUB_P), 
                                                                BABYJUB_P), BABYJUB_P), BABYJUB_P);
        power = addmod(power, mulmod(vars.Z_beta_4, BABYJUB_P - cm.gamma4[0], BABYJUB_P), BABYJUB_P);
        // power = addmod(power, mulmod(Z_beta_4, BABYJUB_P - addmod(addmod(gamma4[0], 
        //                                                                         mulmod(gamma4[1], z, BABYJUB_P), 
        //                                                                         BABYJUB_P),
        //                                                                         mulmod(gamma4[2], 
        //                                                                         mulmod(z, z, 
        //                                                                         BABYJUB_P), 
        //                                                         BABYJUB_P),
        //                                                     BABYJUB_P),
        //                             BABYJUB_P), BABYJUB_P);
        power = addmod(power, mulmod(Z_beta_5, BABYJUB_P - cm.gamma5[0], BABYJUB_P), BABYJUB_P);
        power = addmod(power, mulmod(Z_beta_6, BABYJUB_P - addmod(cm.gamma6[0], mulmod(cm.gamma6[1], z, BABYJUB_P), 
                                                                BABYJUB_P), BABYJUB_P), BABYJUB_P);
        power = addmod(power, mulmod(Z_beta_7, BABYJUB_P - cm.gamma7[0], BABYJUB_P), BABYJUB_P);
        // calculate the PI product
        Pairing.G1Point memory RR = Pairing.mulScalar(Pairing.G1Point(1, 2), power);

        // then add the first item before the uppercase Pi product
        //g^p[x]·zT[z]
        RR = Pairing.plus(RR, Pairing.negate(Pairing.plus(cm.pi_1,
                                                        Pairing.mulScalar(Pairing.G1Point(1, 2),
                                                                        z_calculation(9)))));
        //g^z·w[x]
        RR = Pairing.plus(RR, Pairing.mulScalar(cm.pi_2, z));
        

        // H = (20077374419706392512429686799679330166706149542289304718417735408826647662713, 
        //     14885977252123072602720243771283406369571578426117217829155383966924599215549)
        // RR = (5854638644333843355810549205783961330813896657133136761631324628019915576985, 
        //     10388050893327881316839543061910361629748564725587001630616572947848175252412)
        // check the equation, then check others
        bool result = Pairing.pairing_3point(
            H,
            SRS_G2_1,
            RR,
            t_hxdmax,
            Pairing.negate(cm.pi_2),
            t_hxdmaxplusone
            );
            // && recover(cm.message, cm.sig) == cm.addr //verifySignature
            // && cm.r_1 == addmod(cm.r_tilde, mulmod(d, z_n, BABYJUB_P), BABYJUB_P)
            // && cm.t == addmod(mulmod(cm.r_1, 
            //                       addmod(cm.r_2,
            //                              cm.s_tilde, BABYJUB_P), BABYJUB_P),
            //                 (BABYJUB_P - cm.k), BABYJUB_P)
            // && cm.s_1_tilde == cm.s_2_tilde;

        // temporary code for estimating gas cost, the above is correct version
        // bool result = Pairing.pairing_3point(
        //     H,
        //     SRS_G2_1,
        //     RR,
        //     t_hxdmax,
        //     Pairing.negate(cm.pi_2),
        //     t_hxdmaxplusone
        //     );
        // result = recover(cm.message, cm.sig) == cm.addr; //verifySignature
        // result = cm.r_1 == addmod(cm.r_tilde, mulmod(d, z_n, BABYJUB_P), BABYJUB_P);
        // result = cm.t == addmod(mulmod(cm.r_1, 
        //                           addmod(cm.r_2,
        //                                  cm.s_tilde, BABYJUB_P), BABYJUB_P),
        //                     (BABYJUB_P - cm.k), BABYJUB_P);
        // result = cm.s_1_tilde == cm.s_2_tilde;

        emit verifyResult(result);
        // emit checkData(H.X);
        
        return result;
    }
    

    function z_calculation (uint256 i)
                            internal view returns (uint256){
        
        uint256 result = 1;
        if (i != 1){
            result = mulmod(result, addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 2){
            result = mulmod(result, addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 3){
            result = mulmod(mulmod(result, addmod(z, BABYJUB_P - [z, yz][1], BABYJUB_P), BABYJUB_P)
                            , addmod(z, BABYJUB_P - [z, yz][0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 4){
            result = mulmod(result, addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
            // result = mulmod(mulmod(mulmod(result, addmod(z, BABYJUB_P - [z][2], BABYJUB_P), BABYJUB_P)
            //                 , addmod(z, BABYJUB_P - [z][1], BABYJUB_P), BABYJUB_P)
            //                 , addmod(z, BABYJUB_P - [z][0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 5){
            result = mulmod(result, addmod(z, BABYJUB_P - [y][0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 6){
            result = mulmod(mulmod(result, addmod(z, BABYJUB_P - [z, 1][1], BABYJUB_P), BABYJUB_P)
                            , addmod(z, BABYJUB_P - [z, 1][0], BABYJUB_P), BABYJUB_P);
        }
        if (i != 7){
            result = mulmod(result, addmod(z, BABYJUB_P - [y][0], BABYJUB_P), BABYJUB_P);
        }
        return result;
    }

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
    

}