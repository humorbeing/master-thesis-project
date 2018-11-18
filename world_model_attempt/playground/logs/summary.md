# First 5 vr loss log
- check seed: run same code twice. on seed=1.
- S_t is not involved in training S_t+1.
    - with or without torch.no_grad().
- these are all fresh models.

# moved onto RNN
- first, combined with VAE. so trained V+R model
    - It did not work. everything is worse.
- Only included rnn.params in optimizer, given the similar
result with rnn alone model, but with huge pred_loss.

# ideas 
- [x] change log name as date + content: since i am taking a note in log files.
- [x] realized S_t need it's gradient to train VAE, not only S_t+1
    - [ ] need S_t kl and r loss too.
- at this point, i think pred loss is dominating whole training process
- should i tank note here?
- make a alone rnn, that gives exact result as rnn_me (check math)
- seperate optimizer
- rv model, train with one loss at a time (kl -> kl + recon -> kl + recon + mdn ->...)
- rv model, more weight to RNN model, if competing , then RNN = 10 VAE


# NOTE
- logs/20181119-07-14-30-vr_loss.txt
    - this is joint training.
pred loss is 4300 to 236, and everything else is messed up?
i think pred loss is make too big of a mess.
    - first, make a rnn alone structure in VR, that gives similar result as RNN_me
- logs/20181119-07-37-27-vr_loss.txt
    - reprecate result as in rnn_me
    - mdn is almost same.
    - look like I messed up pred loss.
    - problem is prediction loss is way too high
        - in rnn_me, it's 0.445
        - in vr_rnn along, it's 19095
        - why? looks like a math problem.
- logs/20181119-08-15-43-vr_loss_alone_rnn.txt
    - add rnn._init(), same result
    - pred loss is too big
    - want to make single dataset