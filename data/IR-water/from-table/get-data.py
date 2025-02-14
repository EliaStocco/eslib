import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------#
# Imaginary refractive index
df = pd.read_csv("raw/imaginary-refractive-index.csv")

# Create an empty DataFrame with specified columns
data = pd.DataFrame(columns=["cm-1", "k"])

J = np.arange(17)

for n, row in df.iterrows():
    tmp = np.asarray(row.iloc[3:])  # Extract absorption values
    tmp = tmp.astype(float)
    # tmp = np.asarray([float(str(int(x))[0] + "." + str(int(x))[1:]) if not np.isnan(x) else np.nan for x in tmp])
    ii = np.logical_not(np.isnan(tmp))  # Mask to remove NaNs
    

    # Compute frequencies
    frequencies = float(row["cm-1"]) - 15798.002 / 16384. * J[ii] * np.power(2, row["XE"])
    frequencies = np.asarray(frequencies)

    assert np.all(frequencies > 0), "error"

    # Compute molar absorption values
    y = tmp[ii] / np.power(10, -row["YE"])

    # Append to DataFrame
    new_row = pd.DataFrame({"cm-1": frequencies, "k": y})
    # Concatenate the new row with the existing DataFrame
    data = pd.concat([data, new_row], ignore_index=True)
    
data = data.sort_values(by="cm-1", ascending=True).reset_index(drop=True)
# Save the processed data to a CSV file
data.to_csv("imaginary-refractive-index.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(6,3))

# Linear plot
axs[0].plot(data["cm-1"], data["k"],color="blue",linewidth=1)
axs[0].set_xlim(0,5000)
axs[0].set_ylim(0,None)
axs[0].set_xlabel('frequency [cm$^{-1}$]')
axs[0].set_ylabel('imag. refractive index')
axs[0].grid(True)

# Log-log plot
axs[1].plot(data["cm-1"], data["k"], color="red",linewidth=1)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlim(1,1.5e4)
axs[1].set_ylim(2e-2,None)
axs[1].set_xlabel('frequency [cm$^{-1}$]')
# axs[1].set_ylabel('absorption [L/mol cm]')
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/imaginary-refractive-index.pdf",bbox_inches='tight')
plt.savefig("images/imaginary-refractive-index.png",dpi=300,bbox_inches='tight')
plt.close()


#------------------------------------------------#
# Real refractive index
df = pd.read_csv("raw/real-refractive-index.csv")

# Create an empty DataFrame with specified columns
data = pd.DataFrame(columns=["cm-1", "n"])

J = np.arange(17)

for n, row in df.iterrows():
    tmp = np.asarray(row.iloc[2:])  # Extract absorption values
    tmp = tmp.astype(float)
    tmp = np.asarray([float(str(int(x))[0] + "." + str(int(x))[1:]) if not np.isnan(x) else np.nan for x in tmp])
    ii = np.logical_not(np.isnan(tmp))  # Mask to remove NaNs
    

    # Compute frequencies
    frequencies = float(row["cm-1"]) - 15798.002 / 16384. * J[ii] * np.power(2, row["XE"])
    frequencies = np.asarray(frequencies)

    assert np.all(frequencies > 0), "error"

    # Compute molar absorption values
    y = tmp[ii]

    # Append to DataFrame
    new_row = pd.DataFrame({"cm-1": frequencies, "n": y})
    # Concatenate the new row with the existing DataFrame
    data = pd.concat([data, new_row], ignore_index=True)
    
data = data.sort_values(by="cm-1", ascending=True).reset_index(drop=True)
# Save the processed data to a CSV file
data.to_csv("real-refractive-index.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(6,3))

# Linear plot
axs[0].plot(data["cm-1"], data["n"],color="blue",linewidth=1)
# axs[0].set_xlim(0,5000)
# axs[0].set_ylim(0,None)
axs[0].set_xlabel(r'frequency [cm$^{-1}$]')
axs[0].set_ylabel('real refractive index')
axs[0].grid(True)

# Log-log plot
axs[1].plot(data["cm-1"], data["n"], color="red",linewidth=1)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
# axs[1].set_xlim(1,1.5e4)
# axs[1].set_ylim(2e-2,None)
axs[1].set_xlabel(r'frequency [cm$^{-1}$]')
# axs[1].set_ylabel('absorption [L/mol cm]')
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/real-refractive-index.pdf",bbox_inches='tight')
plt.savefig("images/real-refractive-index.png",dpi=300,bbox_inches='tight')
plt.close()

#------------------------------------------------#
# Molar absorption coefficient
df = pd.read_csv("raw/molar-absorption.csv")

# Create an empty DataFrame with specified columns
data = pd.DataFrame(columns=["cm-1", "mol-abs"])

J = np.arange(17)

for n, row in df.iterrows():
    tmp = np.asarray(row.iloc[3:])  # Extract absorption values
    ii = np.logical_not(np.isnan(tmp))  # Mask to remove NaNs

    # Compute frequencies
    frequencies = row["cm-1"] - 15798.002 / 16384 * J[ii] * np.power(2, row["XE"])
    frequencies = np.asarray(frequencies)

    assert np.all(frequencies > 0), "error"

    # Compute molar absorption values
    y = tmp[ii] * np.power(10, row["YE"])

    # Append to DataFrame
    new_row = pd.DataFrame({"cm-1": frequencies, "mol-abs": y})
    # Concatenate the new row with the existing DataFrame
    data = pd.concat([data, new_row], ignore_index=True)
    
data = data.sort_values(by="cm-1", ascending=True).reset_index(drop=True)
# Save the processed data to a CSV file
data.to_csv("molar-absorption.csv", index=False)

# https://en.wikipedia.org/wiki/Molar_absorption_coefficient
volumetric_density_25 = 997.13 #g/L at 25 celsius
molar_density = 18.0146 # g/mole
factor = volumetric_density_25 / molar_density
data["mol-abs"] *= factor 
# https://media.iupac.org/publications/analytical_compendium/Cha11sec2.pdf
data["mol-abs"] *= np.log(10) # Naperian
data = data.rename(columns={'mol-abs': 'absorption'})
data.to_csv("absorption.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(6,3))

# Linear plot
axs[0].plot(data["cm-1"], data["absorption"]/1000,color="blue",linewidth=1)
axs[0].set_xlim(0,5000)
axs[0].set_ylim(0,None)
axs[0].set_xlabel('frequency [cm$^{-1}$]')
axs[0].set_ylabel(r'absorption [10$^{-3}$ cm$^{-1}$]')
axs[0].grid(True)

# Log-log plot
axs[1].plot(data["cm-1"], data["absorption"]/1000, color="red",linewidth=1)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlim(1,1.5e4)
axs[1].set_ylim(2e-2,None)
axs[1].set_xlabel('frequency [cm$^{-1}$]')
# axs[1].set_ylabel('absorption [L/mol cm]')
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/absorption.pdf",bbox_inches='tight')
plt.savefig("images/absorption.png",dpi=300,bbox_inches='tight')
plt.close()


#------------------------------------------------#
alpha = pd.read_csv("absorption.csv")
refractive = pd.read_csv("real-refractive-index.csv") 

x = alpha["cm-1"]
y = alpha["absorption"] * refractive["n"]/1000

data = pd.DataFrame(data={"cm-1":x,"alpha-n":y})
data.to_csv("spectrum.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(6,3))

# Linear plot
axs[0].plot(x,y,color="blue",linewidth=1)
axs[0].set_xlim(0,5000)
axs[0].set_ylim(0,None)
axs[0].set_xlabel('frequency [cm$^{-1}$]')
axs[0].set_ylabel(r'$\alpha\left(\nu\right)n\left(\nu\right)$ [10$^{-3}$ cm$^{-1}$]')
axs[0].grid(True)

# Log-log plot
axs[1].plot(x,y, color="red",linewidth=1)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlim(1,1.5e4)
axs[1].set_ylim(2e-2,None)
axs[1].set_xlabel('frequency [cm$^{-1}$]')
# axs[1].set_ylabel('absorption [L/mol cm]')
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/spectrum.pdf",bbox_inches='tight')
plt.savefig("images/spectrum.png",dpi=300,bbox_inches='tight')
plt.close()
