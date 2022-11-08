const Verifier = artifacts.require("Verifier");
// const PolyCoeff = artifacts.require("PolyCoeff");

module.exports = function (deployer) {
  // deployer.deploy(PolyCoeff);
  deployer.deploy(Verifier);
};
