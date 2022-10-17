const Constants = artifacts.require("Constants");

module.exports = function (deployer) {
  deployer.deploy(Constants);
};
