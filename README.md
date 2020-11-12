# gym.cartpole-v0

#### Project Structure:

#### Implementations

DQN: deep Q network
Policy iteration

#### solved using off-policy DQN:
## optimize using bellman equation: $Q(s,a)=r + \gamma \underset{a'\sim A}{max} Q(s',a')$
## optimal form $ q(s,a;\theta^*)= E[Q(s,a)]$

## vanilla DQN algorithm pseudocode
## init Q network: $q(s,a;\theta_1)$
## init target Q network: $q(s',a';\theta_2), \theta_2 = \theta_1$
## for every state transition: $Loss=\frac{1}{N}\sum \bigg(J(q(s,a; \theta_1), (r + \gamma \underset{a'\sim A}{max} q(s',a';\theta_2)(1-done))\bigg)$
## $\theta_1 \leftarrow Optimize(Loss, \theta_1)$
## every once in a while: $\theta_2 \leftarrow \theta_1$

#### solved using policy iteration:
* reinforcement learning algorithm: approximate policy iteration
* deep neural network: input(4)-dense(128)-drop(0.8)-dense(256)-drop(0.8)-dense(512)-drop(0.8)-dense(256)-drop(0.8)-dense(128)-softmax(2), opt=adam, lr=1e-3

```python
# core algorithm: policy iteration (src: /experiment.ipynb)

def policyIteration(epochs, exploration, batch_size=10):
    avg_rwds = []
    for i in range(epochs):
        # evaluate the average reward of 10 games
        rwd, avg_act = evaluate(env, learner, 5, render=True)
        
        # report current agent performance (average reward)
        avg_rwd = np.mean(rwd)
        avg_rwds.append(avg_rwd)
        print(f"iteration {i} avg_rwd {avg_rwd}")
        
        # create training data with random exploration rate
        data, acc_rwd, rwd = generateDataBatch(env, learner, size=batch_size, exploration=exploration, criteria=avg_rwd)
        
        # train the policy network with 3 epochs
        learner.train(DATA=data, epochs=3, verbose=2)
        clear_output(wait=True)
        
    return avg_rwds
```

