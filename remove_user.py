import numpy as np
import os

db_path = "db/embeddings.npz"

if not os.path.exists(db_path):
    print("No database found.")
    exit()

data = np.load(db_path, allow_pickle=True)
names = list(data["names"])
embeddings = list(data["embeddings"])

if len(names) == 0:
    print("No users registered.")
    exit()

print("\nRegistered Users:")
for i, name in enumerate(names):
    print(f"{i+1}. {name}")

choice = input("\nEnter user number or name to delete: ").strip()

index = -1

if choice.isdigit():
    idx = int(choice) - 1
    if 0 <= idx < len(names):
        index = idx
else:
    if choice in names:
        index = names.index(choice)

if index == -1:
    print("Invalid selection.")
    exit()

removed_name = names.pop(index)
embeddings.pop(index)

np.savez(db_path, names=names, embeddings=embeddings)

print(f"User '{removed_name}' removed successfully.")
