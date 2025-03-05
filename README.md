# Hyperspectral Image Classification

## 📌 Project Overview
This project focuses on **hyperspectral image classification**, leveraging deep learning and machine learning techniques to analyze hyperspectral images and classify them into different categories. The dataset used includes hyperspectral images from the **Indian Pines dataset**.

## 📂 Folder Structure
```
📂 Project Root
│-- 📂 env/              # Virtual environment (ignored in Git)
│-- 📂 input/            # Input hyperspectral images and datasets
│-- 📂 output/           # Model outputs, classification results
│-- 📂 papers/           # Research papers and references
│-- 📂 report/           # Project reports and documentation
│-- 📂 src/              # Source code files
│   │-- train.py         # Training script
│   │-- test.py          # Testing script
│-- .gitignore          # Git ignore file
│-- README.md           # Project documentation (this file)
```

## 🚀 Installation & Setup
To set up the project on your local machine:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```sh
   cd your-repo-name
   ```
3. Create and activate a virtual environment:
   ```sh
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\activate     # On Windows
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 🛠 Usage
### **Training the Model**
Run the training script:
```sh
python src/train.py
```

### **Testing the Model**
Run the test script:
```sh
python src/test.py
```

## 📊 Results
The classification results, model checkpoints, and preprocessed data are stored in the `output/` folder. The classification accuracy achieved is **97%**.

## 📄 Research References
- [A Fast and Compact 3-D CNN for Hyperspectral Image Classification](#)
- [SANet: A Self-Attention Neural Network for Hyperspectral Image Classification](#)

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## 📜 License
This project is licensed under the **MIT License**.

