// Modified from https://github.com/appliedzkp/semaphore/blob/master/contracts/sol/verifier.sol
// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;
pragma experimental ABIEncoderV2;
import "./Pairing.sol";
import { Constants } from "./Constants.sol";

contract Verifier is Constants {

    using Pairing for *;

    // The G1 generator
    Pairing.G1Point SRS_G1_0 = Pairing.G1Point({
        X: Constants.SRS_G1_X_Pos[0],
        Y: Constants.SRS_G1_Y_Pos[0]
    });

    // The G2 generator
    Pairing.G2Point g2Generator = Pairing.G2Point({
        X: [ Constants.SRS_G2_X_0_Pos[0], Constants.SRS_G2_X_1_Pos[0] ],
        Y: [ Constants.SRS_G2_Y_0_Pos[0], Constants.SRS_G2_Y_1_Pos[0] ]

    });

    Pairing.G2Point SRS_G2_hAlphaX1 = Pairing.G2Point({
        X: [ Constants.SRS_G2_X_0_Pos[1], Constants.SRS_G2_X_1_Pos[1] ],
        Y: [ Constants.SRS_G2_Y_0_Pos[1], Constants.SRS_G2_Y_1_Pos[1] ]
    });

    Pairing.G2Point SRS_G2_hAlphaX0 = Pairing.G2Point({
        X: [ Constants.SRS_G2_X_0_Pos[2], Constants.SRS_G2_X_1_Pos[2] ],
        Y: [ Constants.SRS_G2_Y_0_Pos[2], Constants.SRS_G2_Y_1_Pos[2] ]
    });

    Pairing.G2Point SRS_G2_hAlphaXdMax = Pairing.G2Point({
        X: [ Constants.SRS_G2_X_0_Pos[3], Constants.SRS_G2_X_1_Pos[3] ],
        Y: [ Constants.SRS_G2_Y_0_Pos[3], Constants.SRS_G2_Y_1_Pos[3] ]
    });

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
     * @param _hAlphaX1 h^{alpha*x}
     * @param _hAlphaX0 h^{alpha}
     * @param _hAlphaXdMax h^{x^(-d+max)}
     */
    function verify(
        Pairing.G1Point memory _commitment, // F
        Pairing.G1Point memory _proof, // W
        uint256 _index,  // z
        uint256 _value  // F(z) or v
    ) public view returns (bool) {
        _commitment = Pairing.G1Point({
            X: 231014479782015245234485944771714376129960099757713828634132041508962534993,
            Y: 15099064614159299663080487230823884942440586700446010397106703856370686471325
        }); // F
        _proof = Pairing.G1Point({
            X: 14274528744475051507926390565853859775009032673903686347573470949646881561122,
            Y: 11381013419076667208658291680033584733431771688109725038119192064209014907614
        }); // W
        _index = 7282838590790841256333066252499723692377153823335888511276159981254439234130;  // z
        _value =14960434337026191661336017507630746399921995025384487953384274040134774205633; // F(z) or v
        // Make sure each parameter is less than the prime q
        require(_commitment.X < BABYJUB_P, "Verifier.verifyKZG: _commitment.X is out of range");
        require(_commitment.Y < BABYJUB_P, "Verifier.verifyKZG: _commitment.Y is out of range");
        require(_proof.X < BABYJUB_P, "Verifier.verifyKZG: _proof.X is out of range");
        require(_proof.Y < BABYJUB_P, "Verifier.verifyKZG: _proof.Y is out of range");
        require(_index < BABYJUB_P, "Verifier.verifyKZG: _index is out of range");
        require(_value < BABYJUB_P, "Verifier.verifyKZG: _value is out of range");
       
        Pairing.G1Point memory negProof = Pairing.negate(Pairing.mulScalar(_proof, _index));
        Pairing.G1Point memory mulProof = Pairing.MulPoint(Pairing.mulScalar(SRS_G1_0, _value), negProof);
        Pairing.G1Point memory negCm = Pairing.negate(_commitment);

        return Pairing.pairing(
            _proof,
            SRS_G2_hAlphaX1,
            mulProof,
            SRS_G2_hAlphaX0,
            negCm,
            SRS_G2_hAlphaXdMax
        );
        // return false;
    }

    function verifySonic(
        uint256[21] memory Proof,
        uint256[3] memory Randoms
    ) public view returns (bool) {
        
        uint256 t = mulmod(addmod(Proof[6], Proof[9], Pairing.PRIME_Q), 
                            addmod(addmod(Proof[12], Proof[15], Pairing.PRIME_Q), 
                                            Proof[20], Pairing.PRIME_Q), Pairing.PRIME_Q) - 1;

        return verify(Pairing.G1Point(Proof[0], Proof[1]), 
                      Pairing.G1Point(Proof[7], Proof[8]),
                      Randoms[1],
                      Proof[6]) &&
                verify(Pairing.G1Point(Proof[0], Proof[1]), 
                      Pairing.G1Point(Proof[13], Proof[14]),
                      Randoms[1],
                      Proof[12]) &&
                verify(Pairing.G1Point(Proof[2], Proof[3]), 
                      Pairing.G1Point(Proof[10], Proof[11]),
                      Randoms[1],
                      Proof[9]) &&
                verify(Pairing.G1Point(Proof[2], Proof[3]), 
                      Pairing.G1Point(Proof[16], Proof[17]),
                      Randoms[1],
                      Proof[15]) &&
                verify(Pairing.G1Point(Proof[4], Proof[5]), 
                      Pairing.G1Point(Proof[18], Proof[19]),
                      Randoms[1],
                      t);
    }

}