# gym.cartpole-v0
solved using approximate policy iteration

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

