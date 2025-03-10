#=
in julia REPL, check:
registry status
if LabRegistry not in list, do:
registry add git@github.com:lemieux-lab/LabRegistry.git
if error, do:
ENV["JULIA_PKG_USE_CLI_GIT"] = true
=#

using DataFrames

# thank you carl!
include("/home/golem/scratch/munozc/DDPM/LINCS_data_explorer/SETUP/Lincs.jl")
using .Lincs: Data

@time data = Data("/home/golem/scratch/munozc/DDPM/Data/out_20M_4.h5")

