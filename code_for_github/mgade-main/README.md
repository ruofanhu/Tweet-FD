# mgade
The original code is for "A Dual-Attention Network for Joint Named Entity Recognition and Sentence Classification of Adverse Drug Events", Findings of EMNLP, 2020. In this work, we use and adapt the code for evaluation. The code we have edited are evaluator.py, experiment.py, and model.py.

Run experiment with 
```cmd
    python experiment.py config_file.conf
```

Below list description of each conf file
* [new_config.conf](new_config.conf)  for TRC + EMD task
* [new_re_config.conf](new_re_config.conf) for TRC + RED task
* [new_re_cls_config.conf](new_re_cls_config.conf) for TRC + ERC task

The original code adapted from "Marek Rei and Anders Søgaard. Jointly learning to label sentences and tokens. AAAI 2019", for binary classification of words and sentences with a single type of attention from the words/entities.


If you find this code or the original work useful, please cite this paper:

@inproceedings{wunnava2020dual,
  title={A Dual-Attention Network for Joint Named Entity Recognition and Sentence Classification of Adverse Drug Events},
  author={Wunnava, Susmitha and Qin, Xiao and Kakar, Tabassum and Kong, Xiangnan and Rundensteiner, Elke},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={3414--3423},
  year={2020}
}
