import os

import numpy as np
import pandas as pd
import spectrums as sp

if __name__ == "__main__":
    keywords = ["cap", "stipe", "gills"]
    folder_path = "data"
    species = []
    spectrums = []
    parts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            S = sp.Spectrum(path=os.path.join(root, file))
            S.select([601, 1800], [2412, 3998])
            wn = S.wavenums
            if len(wn) == 0:
                print(f"{file}: spectrum length = 0")
                continue
            if len(spectrums) > 0:
                if len(wn) != len(spectrums[-1]):
                    print(f"{file}: invalid spectrum length = {len(wn)}")
                    continue
            for word in keywords:
                if word in file:
                    parts.append(word)
                    species.append(root.split("/")[1])
                    spectrums.append(S.data)
    if len(spectrums) == 0:
        raise ValueError("Spectrum parsed data is empty!")
    else:
        spectrum_paths = []
        np.save(file=f"spectra_data/wavenumbers.npy", arr=wn)
        for i in range(len(spectrums)):
            spectrum_path = f"spectra_data/sample_{i}_{species[i]}_{parts[i]}.npy"
            np.save(file=spectrum_path, arr=spectrums[i])
            spectrum_paths.append(spectrum_path)
        metadata = {"spectrum_path": spectrum_paths, "species": species, "part": parts}
        df = pd.DataFrame(metadata)
        df.to_csv("spectra_metadata.csv", encoding="utf-8", index=False, header=True)
