# Solidity on L2 for Staking Contracts

Staking and slashing logic runs in `StakeManager.sol`, a Solidity contract deployed on an Ethereum L2 (Arbitrum/Optimism). The Rust binary interacts with it via `alloy` v2 generated bindings.

**Why on-chain:** Economic security is the foundation of the protocol. Without real stake at risk, there's no Sybil resistance and no meaningful slashing. Reputation-only systems fail because reputation is free to accumulate and discard — a malicious node can attack without losing anything.

**Why L2, not L1:** Gas costs. Flagging a misbehaving node needs to happen frequently and cheaply. On L1 Ethereum, each flag transaction would cost $10-50 at peak, making the protocol uneconomical. L2 (Arbitrum, Optimism) brings flag costs to pennies. Settlement to L1 for finality is still available for large slashing events.

**Why not a custom L1/L2:** Building a new token/chain requires months of audit work, liquidity bootstrapping, and regulatory uncertainty. USDC on existing L2s is battle-tested and immediately usable.

**Why not Solana:** EVM tooling (Hardhat, Slither, MetaMask, WalletConnect, USDC) is far more mature. The team has Ethereum experience.

**V1 scope:** Simulated staking (no real USDC transfers) for the MVP. Real USDC integration in V2 after a professional audit. Slither static analysis is required: zero high-severity findings before deployment.

**Contract architecture:**
- `stake(node, amount)` — lock USDC as collateral
- `flag(node, reporter)` — accumulate misbehavior flags
- `slash(node, flags)` — graduated: 10 flags → freeze 48h, 50 flags → 20% slash + ban
- `unstake(node)` — withdraw remaining stake (subject to cooldown)
