# CTRL-MoE (Contextual Traffic Reinforcement Learning with Mixture of Experts) 

This repository contains experiments and analyses for Contextual Traffic Reinforcement Learning using the Mixture of Experts (MoE) framework.

---

## Prerequisites
- **Python**: Version 3.7 or later.
- **PyTorch**: Ensure PyTorch is installed. [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).
- **SUMO**: Install SUMO and add it to your system's PATH. [SUMO Installation Guide](https://sumo.dlr.de/docs/Installing.html).
- **SUMO-RL**: Install the SUMO-RL library. [GitHub Repository](https://github.com/LucasAlegre/sumo-rl).
- **Stable-Baselines3**: Install Stable-Baselines3. [GitHub Repository](https://github.com/DLR-RM/stable-baselines3).
- **Dependencies**: Install additional Python libraries using the provided `requirements.txt`.

```bash
pip install -r requirements.txt
```

## File Structure

```plaintext
CTRL-MoE/
│
├── nets/                   # Traffic scenarios and SUMO configurations
├── experts/                # Expert policy training and storage
├── models/                 
│   ├── moe/                # Mixture of Experts framework
│   ├── policies/           # Policy-related code for RL
├── envs/                   
│   ├── contexts/           # Context definitions
│   └── sumoenv.py          # SUMO environment for RL
├── utils/                  # Helper scripts for preprocessing and evaluation
├── docs/                   # Documentation and images
├── requirements.txt        # Python dependencies
├── main.py                 # Main script to run the MoE framework
├── LICENSE                 # Project license
└── README.md               # Project README
```


## License
This project is licensed under the MIT License. See the LICENSE file for more details.







