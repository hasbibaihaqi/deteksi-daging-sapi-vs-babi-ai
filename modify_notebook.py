import json

file_path = 'training/train_model.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_cells = [
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "e93a6b8c",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Mulai Training Model\n",
            "epochs = 10\n",
            "\n",
            "history = model.fit(\n",
            "    train_generator,\n",
            "    epochs=epochs,\n",
            "    validation_data=validation_generator\n",
            ")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "f5a28b1a",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "# 8. Simpan Model untuk digunakan di Web App\n",
            "os.makedirs('../models', exist_ok=True)\n",
            "model.save('../models/meat_model.h5')\n",
            "print(\"Model berhasil disimpan di folder 'models'!\")"
        ]
    }
]

notebook['cells'].extend(new_cells)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated.")
