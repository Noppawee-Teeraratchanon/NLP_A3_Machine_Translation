# NLP_A3_Machine_Translation

To run app, 
1. Load the files from this repository
2. Run
```sh
python app/app.py
```
3. Access the app with http://127.0.0.1:5000

Dataset that I used to train: scb_mt_enth_2020 from siam commercial bank (SCB) (https://airesearch.in.th/releases/machine-translation-datasets/), (https://huggingface.co/datasets/scb_mt_enth_2020)

Train and Validation loss graph
General attention:
![Alt Text](https://github.com/Noppawee-Teeraratchanon/NLP_A3_Machine_Translation/blob/main/images/plot_general.png)
Multiplicative attention:
![Alt Text](https://github.com/Noppawee-Teeraratchanon/NLP_A3_Machine_Translation/blob/main/images/plot_multiplicative.png)

Attention map
General attention:
![Alt Text](https://github.com/Noppawee-Teeraratchanon/NLP_A3_Machine_Translation/blob/main/images/attention_maps_general.png)
Multiplicative attention:
![Alt Text](https://github.com/Noppawee-Teeraratchanon/NLP_A3_Machine_Translation/blob/main/images/attention_maps_multi.png)

Screenshot of my web interface:
![Alt Text](https://github.com/Noppawee-Teeraratchanon/NLP_A3_Machine_Translation/blob/main/images/A3_webpage.png)