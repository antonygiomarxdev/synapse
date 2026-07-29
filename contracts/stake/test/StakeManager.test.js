const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("StakeManager", function () {
  async function deployFixture() {
    const [owner, miner] = await ethers.getSigners();
    const StakeManager = await ethers.getContractFactory("StakeManager");
    const stakeManager = await StakeManager.deploy();
    return { stakeManager, owner, miner };
  }

  it("should accept stake", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("100") });
    const info = await stakeManager.stakes(nodeId);
    expect(info.amount).to.equal(ethers.parseEther("100"));
  });

  it("should start reputation at 100", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("10") });
    expect(await stakeManager.getReputation(nodeId)).to.equal(100);
  });

  it("should flag a node", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("100") });
    await stakeManager.connect(miner).flag(nodeId);
    const info = await stakeManager.stakes(nodeId);
    expect(info.flags24h).to.equal(1);
  });

  it("should freeze at 10 flags", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("100") });
    for (let i = 0; i < 10; i++) {
      await stakeManager.connect(miner).flag(nodeId);
    }
    expect(await stakeManager.isFrozen(nodeId)).to.be.true;
  });

  it("should update reputation", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("10") });
    await stakeManager.connect(miner).updateReputation(nodeId, 720);
    expect(await stakeManager.getReputation(nodeId)).to.equal(720);
  });

  it("should freeze and unfreeze via standalone functions", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("100") });
    await stakeManager.connect(miner).freeze(nodeId, 3600);
    expect(await stakeManager.isFrozen(nodeId)).to.be.true;
    await stakeManager.connect(miner).unfreeze(nodeId);
    expect(await stakeManager.isFrozen(nodeId)).to.be.false;
  });

  it("should ban and unban", async function () {
    const { stakeManager, miner } = await deployFixture();
    const nodeId = ethers.zeroPadValue(miner.address, 32);
    await stakeManager.connect(miner).stake(nodeId, { value: ethers.parseEther("100") });
    await stakeManager.connect(miner).ban(nodeId);
    expect(await stakeManager.isBanned(nodeId)).to.be.true;
    await stakeManager.connect(miner).unban(nodeId);
    expect(await stakeManager.isBanned(nodeId)).to.be.false;
  });
});
