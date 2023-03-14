import numpy as np
import matplotlib.pyplot as plt
from pyiron import Project
from pymatgen.ext.matproj import MPRester

# Replace this with your Materials Project API key
api_key = "your_api_key_here"

# Define the Materials Project IDs for the 4 materials
material_ids = ["mid1", "mid2", "mid3", "mid4"]

# Create a pyiron project
project_name = "convergence_study"
pr = Project(project_name)

# Define k-point grid range and ENCUT range for the convergence study
kpoints_range = np.arange(2, 12, 2)
encut_range = np.arange(400, 1201, 200)

# Create a reference job to base the convergence study on
ref_job = pr.create.job.Vasp("ref_job")
ref_job.exchange_correlation_functional = "PBE"

# Load the structures from the Materials Project
with MPRester(api_key) as mpr:
    for material_id in material_ids:
        structure = mpr.get_structure_by_material_id(material_id)
        pyiron_structure = pr.create.structure.structure(pymatgen_structure=structure)

        # Create a job for each k-point and ENCUT combination
        for k in kpoints_range:
            for encut in encut_range:
                job_name = f"{material_id}_vasp_k{k}_encut{encut}"
                job = pr.create.job.Vasp(job_name, delete_existing_job=True)
                job.structure = pyiron_structure
                job.exchange_correlation_functional = "PBE"
                job.set_encut(encut)
                job.set_kpoints(mesh=[k, k, k])
                job.run()

        # Extract total energies for each k-point and ENCUT combination
        total_energies = np.zeros((len(kpoints_range), len(encut_range)))
        
        for i, k in enumerate(kpoints_range):
            for j, encut in enumerate(encut_range):
                job_name = f"{material_id}_vasp_k{k}_encut{encut}"
                job = pr.inspect(job_name)
                total_energies[i, j] = job["output/generic/energy_tot"][-1]

        # Calculate the energy delta relative to the maximum k-point and ENCUT value
        reference_energy = total_energies[-1, -1]
        energy_delta = total_energies - reference_energy

        # Plot the convergence with respect to k-points and ENCUT for each material
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(encut_range, kpoints_range, energy_delta, shading="auto", cmap="viridis")
        plt.colorbar(label="Energy Delta (eV)")
        plt.xlabel("ENCUT (eV)")
        plt.ylabel("K-points (grid size)")
        plt.title(f"Energy Delta with K-points and ENCUT for Material {material_id}")
        plt.show()
