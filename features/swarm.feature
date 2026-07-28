Feature: Speculative Swarm Consensus
  As a Synapse client
  I want reliable, verified inference responses
  So that I can trust the swarm output without reading individual node results

  Background:
    Given a swarm of 5 nodes running Kimi K2.7 Code
    And all nodes have identical model weights (SHA256 verified)
    And each node uses a different random seed

  Scenario: Majority consensus produces correct output
    When I send a prompt "Write a Python function to sort a list"
    Then at least 3 out of 5 nodes produce the same token sequence
    And the coordinator returns the majority result
    And minority nodes are flagged for re-sync

  Scenario: Divergent node is expelled after repeated failures
    Given Node 3 produces divergent tokens for 3 consecutive tokens in one request
    When the divergence counter reaches 3
    Then Node 3 is expelled from the current request
    And Node 3 receives a reputation flag
    And Node 3 does not receive payment for the expelled tokens

  Scenario: Node failure during speculative generation
    Given 5 nodes are processing a realtime request
    When Node 4 disconnects mid-generation
    Then the coordinator continues with 4 remaining nodes
    And consensus requires 3/4 majority
    And the client receives a normal response without errors

  Scenario: Chronic diverger is slashed
    Given Node 5 has received 10 flags in a 24-hour window
    When the 10th flag is registered
    Then Node 5's stake is frozen for 48 hours
    And the gateway stops routing requests to Node 5

  Scenario: Swarm size determines consensus strength
    Given a swarm_size of 3
    Then consensus requires 2/3 majority
    Given a swarm_size of 8
    Then consensus requires 5/8 majority
    And the cost to attack the swarm is $2,500 at $500 stake per node

---

Feature: Swarm DAG Batch Processing
  As a CI/CD pipeline
  I want to process thousands of inference requests simultaneously
  So that I can analyze entire codebases efficiently

  Background:
    Given a swarm of 4 nodes holding Mixtral 8x7B experts
    And each node holds 2 experts
    And expert replicas exist for fault tolerance

  Scenario: Batch routing through expert graph
    When I submit 100 independent requests for code analysis
    Then the gateway assembles the cheapest valid expert route per request
    And multiple requests flow through the expert DAG simultaneously
    And throughput scales with the number of available nodes

  Scenario: Mid-request node failure with replica fallback
    Given Request #42 is using Expert #7 on Node B
    When Node B crashes mid-computation
    Then the gateway re-routes Expert #7 to Node C (replica)
    And Node A (Expert #3) is paid normally for completed work
    And Node B receives no payment and a timeout flag
    And the client receives the complete response without error

  Scenario: Statistical audit catches malicious nodes
    Given a node produces subtly wrong log-probabilities
    When the 5% audit sampling selects that node's response
    Then a second node reruns the same request with seed=0
    And the log-probability matrices diverge beyond threshold
    And the malicious node is slashed

---

Feature: Economic Incentives
  As a miner
  I want fair, transparent payment for verified work
  So that I can earn income from my idle GPU

  Scenario: Miner earns for verified tokens
    Given I serve Expert #3 of Mixtral 8x7B
    And the gateway routes a speculative swarm request through my node
    And my output matches the consensus majority
    Then I receive payment proportional to verified output tokens
    And payment is delivered as USDC on L2

  Scenario: Market maker assembles cheapest route
    Given Expert #3 is available from Miner A at $0.08/1M tokens
    And Expert #7 is available from Miner C at $0.09/1M tokens
    When the gateway assembles a batch route
    Then the cheapest combination (A + C = $0.17) is selected
    And the client pays the catalog price of $0.25/1M tokens

  Scenario: New miners undercut cartel pricing
    Given all current miners of Expert #3 collude to charge $0.50/1M tokens
    When a new miner joins with a $0.12/1M tokens ask price
    Then the gateway immediately routes to the new cheaper miner
    And the cartel miners receive no work until they lower prices

  Scenario: Slashing for free-riding
    Given I register as a miner with $500 stake
    And I return garbage responses without running inference
    When the audit detects my responses diverge from correct results
    Then 20% of my stake is slashed
    And my reputation score resets to 0
    And I am banned if my remaining stake falls below the minimum
