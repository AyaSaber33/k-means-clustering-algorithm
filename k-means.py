import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import random
import math

# ---------------- K-MEANS FROM SCRATCH ---------------- #
class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.clusters = []

    def fit(self, data):
        # Requirement 11: Initial centroids chosen randomly
        if len(data) < self.k:
            return # Avoid error if data is smaller than K
        
        self.centroids = random.sample(data, self.k)

        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]

            # Assign points to nearest centroid
            for point in data:
                distances = [self.euclidean(point, c) for c in self.centroids]
                idx = distances.index(min(distances))
                clusters[idx].append(point)

            # Compute new centroids
            new_centroids = []
            for i in range(self.k):
                if clusters[i]:
                    # Mean of all points in cluster
                    mean = [sum(x)/len(x) for x in zip(*clusters[i])]
                    new_centroids.append(mean)
                else:
                    # Logic fix: Handle empty clusters
                    new_centroids.append(self.centroids[i])

            # Stop if converged
            if new_centroids == self.centroids:
                self.clusters = clusters
                break
            self.centroids = new_centroids
            self.clusters = clusters

    def euclidean(self, p1, p2):
        # Standard Euclidean Distance formula
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# ---------------- OUTLIER DETECTION ---------------- #
def detect_outliers(clusters, centroids, threshold=2.0):
    outliers = []
    for i, cluster in enumerate(clusters):
        if not cluster: continue
        centroid = centroids[i]
        
        distances = [math.sqrt(sum((a-b)**2 for a,b in zip(p, centroid))) for p in cluster]
        avg_dist = sum(distances) / len(distances)
        
        # Standard Deviation for better statistical detection
        variance = sum((d - avg_dist)**2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)

        for point, dist in zip(cluster, distances):
            # Requirement 12: Detect outliers
            # A point is an outlier if it's much further than the average distance
            if dist > (avg_dist + threshold * std_dev):
                outliers.append(point)
    return outliers

# ---------------- GUI ---------------- #
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Mall Customer Clustering Tool")
        self.root.geometry("650x600")
        self.file_path = ""

        # UI Elements
        tk.Label(root, text="K-Means Clustering Analysis", font=('Arial', 14, 'bold')).pack(pady=10)

        tk.Button(root, text="Step 1: Select CSV File", command=self.select_file).pack()
        self.lbl_file = tk.Label(root, text="No file selected", fg="blue")
        self.lbl_file.pack(pady=5)

        tk.Label(root, text="Step 2: Percentage of data to read (0-100):").pack()
        self.perc_entry = tk.Entry(root)
        self.perc_entry.insert(0, "70")
        self.perc_entry.pack()

        tk.Label(root, text="Step 3: Number of Clusters (K):").pack()
        self.k_entry = tk.Entry(root)
        self.k_entry.insert(0, "3")
        self.k_entry.pack()

        tk.Button(root, text="Run Analysis", bg="green", fg="white", font=('Arial', 10, 'bold'), command=self.run).pack(pady=15)

        # Output frame with Scrollbar (The Tip)
        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output = tk.Text(frame, yscrollcommand=self.scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.output.yview)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.lbl_file.config(text=f"Selected: {self.file_path.split('/')[-1]}")

    def run(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file.")
            return

        try:
            # Inputs (Requirement 13.a)
            k = int(self.k_entry.get())
            perc = float(self.perc_entry.get()) / 100.0

            df = pd.read_csv(self.file_path)
            
            # Requirement 9: Slicing by percentage
            n = int(len(df) * perc)
            df = df.head(n)

            # Selecting numeric data (Age, Income, Spending Score)
            numeric_data = df.select_dtypes(include=['number'])
            # We often drop 'CustomerID' for the actual clustering math
            if 'CustomerID' in numeric_data.columns:
                numeric_data = numeric_data.drop(columns=['CustomerID'])
                
            # min max normalization
            numeric_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
            
            data_list = numeric_data.values.tolist()

            # Execute K-Means
            model = KMeans(k=k)
            model.fit(data_list)
            outliers = detect_outliers(model.clusters, model.centroids)

            # Requirement 13.b: Output Results
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Showing results for {n} records...\n")

            for i, cluster in enumerate(model.clusters):
                self.output.insert(tk.END, f"\n{'='*20}\nCLUSTER {i+1} (Points: {len(cluster)})\n{'='*20}\n")
                for point in cluster:
                    self.output.insert(tk.END, f"{point}\n")

            self.output.insert(tk.END, f"\n\n{'!'*20}\nOUTLIER RECORDS\n{'!'*20}\n")
            if outliers:
                for o in outliers:
                    self.output.insert(tk.END, f"{o}\n")
            else:
                self.output.insert(tk.END, "No outliers detected.\n")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input or file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()