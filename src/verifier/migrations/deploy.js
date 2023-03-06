// 引包
const ethers = require("ethers");
const fs = require("fs-extra");

async function main() {
  // 建立ganache网络
  let provider = new ethers.providers.JsonRpcProvider("http://127.0.0.1:7545");
  //用户连接到网络
  let wallet = new ethers.Wallet(
    "d8b3561b40d9a7b2572211ded5e3af249e0e0e3be69e27e37de8241ec43a718e", //私钥
    provider
  );

  // 获取solidity的接口和字节码
  const abi = fs.readFileSync("./SimpleStorage_sol_SimpleStorage.abi", "utf8");
  const binary = fs.readFileSync(
    "./SimpleStorage_sol_SimpleStorage.bin",
    "utf8"
  );
  //用户创建交易
  const contractFactory = new ethers.ContractFactory(abi, binary, wallet);
  console.log("Deploying, please wait...");
  //用户部署智能合约
  const contract = await contractFactory.deploy();
  console.log(contract);

  console.log(`查看项目地址：${contract.address}`);
  // 设置最喜欢的数字
  await contract.store(9);
  // 查看最喜欢的数字
  const storeNumber = await contract.retrieve();
  console.log(`设置的商店最幸运的数字是: ${storeNumber}`);

  //增加一个人
  await contract.addPerson("Daf", 8);
  //查看添加的第一个人的最喜欢的数字
  const { favoriteNumber, name } = await contract.people(0);
  console.log(`${name}最喜欢的数字是：${favoriteNumber}`);

}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });