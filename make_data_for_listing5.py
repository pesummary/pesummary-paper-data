from pesummary.io import write
import numpy as np

# make the PSD files
for IFO in ["H1", "L1", "V1"]:
    frequencies = np.linspace(0, 1024, 1000)
    psd = np.random.uniform(10, 0.1, 1000)
    data = np.vstack([frequencies, psd]).T
    np.savetxt(
        "psd_{}.dat".format(IFO), data, delimiter="\t",
        header="\t".join(["frequencies", "psd"])
    )

parameters = [
    "mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_jl", "phi_12", "psi", "theta_jn", "ra", "dec",
    "luminosity_distance", "geocent_time", "redshift",
    "mass_1_source", "mass_2_source", "log_likelihood"
]
n_samples = 1000
# make the result files
for file_format, extension in zip(["bilby", "lalinference"], ["json", "hdf5"]):
    data = np.array([np.random.random(18) for i in range(n_samples)])
    distance = np.random.random(n_samples) * 500
    mass_1 = np.random.random(n_samples) * 100
    q = np.random.random(n_samples) * 100
    a_1 = np.random.uniform(0, 0.99, n_samples)
    a_2 = np.random.uniform(0, 0.99, n_samples)
    for num, i in enumerate(data):
        data[num][12] = distance[num]
        data[num][0] = mass_1[num]
        data[num][1] = mass_1[num] * q[num]
        data[num][2] = a_1[num]
        data[num][3] = a_2[num]
    write(
        parameters, data, file_format=file_format, overwrite=True,
        filename="test.{}".format(extension),
    )
