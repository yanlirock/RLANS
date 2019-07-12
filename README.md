# RLANS
Learning Adversarial Networks for Semi-Supervised Text Classification via Policy Gradient

Require: Python 3.6, Pytorch 1.0

1)You should split the data into labeled and unlabeled data
```
python split_labeled_unlabeled.py
```


2)You should training the language model
```
python language_model_training.py --cuda --batch_size=32 --lr=0.01 --reduce_rate=0.9 --save='/ag_lm_model/'
```
the "--save" specifies the direction where you want save your pretrained language model. 


3)You can conduct adv training for RLANS model
```
python Adversarial_training.py --cuda --lr=0.001 --batch_size=128 --save='/ag_adv_model/' --pre_train='/ag_lm_model' --number_per_class=1000 --reduce_rate=0.95
```
the "--pre_train" specifies the direction where your pretrained language model has saved.

the "--save" specifies the direction where you want save your RLANS model, and the testing result of each epoch. 


4)For comparison, you can also train a text classification model just with labeled, without specify the pretrained language model
```
python classifier_training.py --cuda --lr=0.001 --batch_size=128 --save='/classify_no_pre/' --pre_train='' --number_per_class=1000 --reduce_rate=0.95
```
the "--save" specifies the direction where you want save the LSTM based text classification model, and the testing result of each epoch. 


4)For comparison, you can train the semisupervised sequence learning model (SSL), without the pretrained language model
```
python classifier_training.py --cuda --lr=0.001 --batch_size=128 --save='/classify_with_pre/' --pre_train='/ag_lm_model' --number_per_class=1000 --reduce_rate=0.95
```
the "--save" specifies the direction where you want save the SSL model, and the testing result of each epoch. 


Reference:

Dai, Andrew M., and Quoc V. Le. "Semi-supervised sequence learning." Advances in neural information processing systems. 2015.

Li, Yan, and Jieping Ye. "Learning adversarial networks for semi-supervised text classification via policy gradient." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018.


