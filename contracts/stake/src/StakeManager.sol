// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.28;

/// @title Synapse Stake Manager
/// @notice Manages miner stakes and slashing on L2.
/// Staking is required to participate as a miner in the swarm.
contract StakeManager {
    // ─── State ──────────────────────────────────────────

    struct StakeInfo {
        uint256 amount;
        uint256 frozenUntil;  // 0 = not frozen
        uint8 flags24h;       // divergence flags in last 24h (rolling)
        uint256 firstFlagAt;  // timestamp of first flag in current 24h window
        uint16 reputation;    // 0-1000
        bool banned;
    }

    mapping(bytes32 => StakeInfo) public stakes;  // nodeId => stake
    mapping(bytes32 => bool) public authorizedMiners;

    uint256 public constant MIN_STAKE = 100 * 1e6; // 100 USDC (6 decimals)
    uint256 public constant MAX_FLAGS_BEFORE_FREEZE = 10;
    uint256 public constant MAX_FLAGS_BEFORE_SLASH = 50;
    uint256 public constant FREEZE_DURATION = 48 hours;
    uint256 public constant SLASH_PERCENT = 20;  // 20%
    uint256 public constant FLAG_WINDOW = 24 hours;

    // ─── Events ─────────────────────────────────────────

    event Staked(bytes32 indexed nodeId, uint256 amount);
    event Unstaked(bytes32 indexed nodeId, uint256 amount);
    event Flagged(bytes32 indexed nodeId, uint8 totalFlags);
    event Slashed(bytes32 indexed nodeId, uint256 amount);
    event Frozen(bytes32 indexed nodeId, uint256 until);
    event Banned(bytes32 indexed nodeId);

    // ─── Modifiers ──────────────────────────────────────

    modifier onlyAuthorized() {
        require(authorizedMiners[bytes32(0)] || msg.sender == address(this), "unauthorized");
        _;
    }

    modifier notBanned(bytes32 nodeId) {
        require(!stakes[nodeId].banned, "node is banned");
        _;
    }

    modifier notFrozen(bytes32 nodeId) {
        require(block.timestamp >= stakes[nodeId].frozenUntil, "stake is frozen");
        _;
    }

    // ─── Core ───────────────────────────────────────────

    function stake(bytes32 nodeId) external payable notBanned(nodeId) {
        require(msg.value >= MIN_STAKE, "below minimum stake");

        stakes[nodeId].amount += msg.value;
        if (stakes[nodeId].reputation == 0) {
            stakes[nodeId].reputation = 100; // Start at Bronze
        }

        emit Staked(nodeId, msg.value);
    }

    function unstake(bytes32 nodeId, uint256 amount)
        external
        notBanned(nodeId)
        notFrozen(nodeId)
    {
        require(stakes[nodeId].amount >= amount, "insufficient stake");
        stakes[nodeId].amount -= amount;
        payable(msg.sender).transfer(amount);
        emit Unstaked(nodeId, amount);
    }

    // ─── Flagging ───────────────────────────────────────

    function flag(bytes32 nodeId) external onlyAuthorized notBanned(nodeId) {
        StakeInfo storage info = stakes[nodeId];

        // Reset rolling window if expired
        if (block.timestamp > info.firstFlagAt + FLAG_WINDOW) {
            info.flags24h = 0;
            info.firstFlagAt = block.timestamp;
        }

        if (info.firstFlagAt == 0) {
            info.firstFlagAt = block.timestamp;
        }

        info.flags24h++;

        emit Flagged(nodeId, info.flags24h);

        // Graduated penalties
        if (info.flags24h >= MAX_FLAGS_BEFORE_SLASH) {
            _slash(nodeId, info);
        } else if (info.flags24h >= MAX_FLAGS_BEFORE_FREEZE) {
            info.frozenUntil = block.timestamp + FREEZE_DURATION;
            emit Frozen(nodeId, info.frozenUntil);
        }
    }

    function _slash(bytes32 nodeId, StakeInfo storage info) internal {
        uint256 slashAmount = (info.amount * SLASH_PERCENT) / 100;
        info.amount -= slashAmount;
        info.flags24h = 0;
        info.reputation = 0;
        info.frozenUntil = 0;

        if (info.amount < MIN_STAKE) {
            info.banned = true;
            emit Banned(nodeId);
        }

        emit Slashed(nodeId, slashAmount);
    }

    // ─── Reputation ─────────────────────────────────────

    function updateReputation(bytes32 nodeId, uint16 newScore)
        external
        onlyAuthorized
        notBanned(nodeId)
    {
        require(newScore <= 1000, "score out of range");
        stakes[nodeId].reputation = newScore;
    }

    function getReputation(bytes32 nodeId) external view returns (uint16) {
        return stakes[nodeId].reputation;
    }

    function getStake(bytes32 nodeId) external view returns (uint256) {
        return stakes[nodeId].amount;
    }
}
