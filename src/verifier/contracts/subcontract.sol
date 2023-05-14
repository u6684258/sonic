pragma solidity >=0.4.22 <0.9.0;

contract Pairing {

    // uint256 yz = mulmod(y, z, BABYJUB_P);
    // if we apply Fiat-Shamir heuristic, then there's no z, y
    uint256 z = uint256(2);
    uint256 y = uint256(3);
    uint256 beta = uint256(1);

    // number of constraints?
    uint256 N = 232;
    // uint256 z_n = expMod(z, N, BABYJUB_P);


}