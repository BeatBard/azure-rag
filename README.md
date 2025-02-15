# Wine Recommendation System using FAISS and LLaMA_CPP

This project is a **Wine Recommendation System** that uses **FAISS** for fast similarity searches and **LLaMA_CPP** for AI-powered responses. The system allows users to query for the best wines based on region, variety, or rating, and it provides AI-generated responses using retrieved wine data.

## 🚀 Features
- **FAISS-based Vector Search** for retrieving wine information quickly.
- **LLaMA_CPP-powered AI Responses** for natural language understanding.
- **FastAPI Backend** to serve API requests.
- **Jupyter Notebook** for debugging and testing the model.
- **Support for Multiple Platforms** (Windows, Linux, macOS).

## 📌 Prerequisites
Before installing and running this project, ensure you have:
- **Python 3.8+** installed
- **Git** installed
- **pip** and **venv** installed for managing dependencies
- **A compatible LLaMA_CPP model** downloaded

## 💾 Installation

### **1️⃣ Clone the Repository**
```sh
<<<<<<< HEAD
git clone https://github.com/your-github-username/wine-recommendation-system.git
=======
git clone https://github.com/BeatBard/azure-rag.git
>>>>>>> b79a5436afa24e4ed561b099f28039a0ba9aa7b8
cd wine-recommendation-system
```

### **2️⃣ Create and Activate a Virtual Environment**
#### **Windows (cmd/PowerShell)**
```sh
python -m venv .venv
.venv\Scripts\activate
```
#### **Linux/macOS**
```sh
python3 -m venv .venv
source .venv/bin/activate
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Download and Setup LLaMA_CPP**
LLaMA_CPP is required to generate responses. Follow these steps to download and set it up:

#### **Windows**
1. **Download LLaMA_CPP from the official repository:**  
   🔗 [LLaMA_CPP Download](https://github.com/Mozilla-Ocho/llamafile)
2. Extract the files and navigate to the directory in your terminal.
3. Run the server:
   ```sh
   llamafile.exe --model llama-7b.gguf --host 127.0.0.1 --port 8080 --n-gpu-layers 20 --ctx-size 2048 --batch-size 8
   ```

#### **Linux/macOS**
1. Install dependencies:
   ```sh
   sudo apt install build-essential cmake python3-pip
   ```
2. Clone and build LLaMA_CPP:
   ```sh
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make
   ```
3. Run LLaMA_CPP:
   ```sh
   ./llamafile --model llama-7b.gguf --host 127.0.0.1 --port 8080 --n-gpu-layers 20 --ctx-size 2048 --batch-size 8
   ```

## 🚀 Running the Wine Recommendation System

### **1️⃣ Start the FastAPI Server**
```sh
uvicorn main:app --host 127.0.0.1 --port 800
```

### **2️⃣ Open API Docs** (Swagger UI)
Visit **[http://127.0.0.1:800/docs](http://127.0.0.1:800/docs)** in your browser.

### **3️⃣ Test a Query**
Send a POST request to `/ask` with a JSON body:
```json
{
    "query": "Best Malbec wine from Argentina"
}
```

## 🎯 Expected Response
The API will return a recommendation like this:
```json
{
    "response": "Alta Vista Alto 2005 is a premium Malbec from Argentina, known for its deep fruit flavors and balanced acidity."
}
```

## 🛠 Debugging in Jupyter Notebook
If you want to test or debug in **Jupyter Notebook**, run:
```sh
jupyter notebook
```
Open `test.ipynb` and execute the cells to troubleshoot the system.

## 💡 Additional Notes
- If LLaMA_CPP **crashes or times out**, restart it with optimized settings (`--ctx-size 2048 --n-gpu-layers 20`).
- If **FastAPI doesn’t start**, check for syntax errors in `main.py`.

## 📜 License
This project is **open-source** under the **MIT License**.

---
🚀 **Now you're all set to explore the world of AI-powered wine recommendations!** 🍷

