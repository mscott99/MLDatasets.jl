function __init__celeba()
    DEPNAME = "CELEBA"

    register(ManualDataDep(
        DEPNAME,
        """
        Dataset: The CELEBA dataset
        Authors: Ziwei Liu   Ping Luo   Xiaogang Wang   Xiaoou Tang 
        Website: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

        CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.

        """
        #"c4a38c50a1bc5f3a1c5537f"#,
        #post_fetch_method=DataDeps.unpack
    ))
end

"""
    CIFAR10(; Tx=Float32, split=:train, dir=nothing)
    CIFAR10([Tx, split])

The CIFAR10 dataset is a labeled subsets of the 80
million tiny images dataset. It consists of 60000
32x32 colour images in 10 classes, with 6000 images
per class.

# Arguments

$ARGUMENTS_SUPERVISED_ARRAY
- `split`: selects the data partition. Can take the values `:train` or `:test`. 

# Fields

$FIELDS_SUPERVISED_ARRAY
- `split`.

# Methods

$METHODS_SUPERVISED_ARRAY
- [`convert2image`](@ref) converts features to `RGB` images.

# Examples

```julia-repl
julia> using MLDatasets: CIFAR10

julia> dataset = CIFAR10()
CIFAR10:
  metadata    =>    Dict{String, Any} with 2 entries
  split       =>    :train
  features    =>    32×32×3×50000 Array{Float32, 4}
  targets     =>    50000-element Vector{Int64}

julia> dataset[1:5].targets
5-element Vector{Int64}:
 6
 9
 9
 4
 1

julia> X, y = dataset[:];

julia> dataset = CIFAR10(Tx=Float64, split=:test)
CIFAR10:
  metadata    =>    Dict{String, Any} with 2 entries
  split       =>    :test
  features    =>    32×32×3×10000 Array{Float64, 4}
  targets     =>    10000-element Vector{Int64}

julia> dataset.metadata
Dict{String, Any} with 2 entries:
  "n_observations" => 10000
  "class_names"    => ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```
"""
struct CELEBA{T} <: SupervisedDataset
    #metadata::Dict{String,Any}
    split::Symbol
    features::T
    #targets::Vector{Int}
end

#CELEBA(; split=:train, Tx=Float32, dir=nothing) = CELEBA(Tx, split; dir)
#CELEBA(split::Symbol; kws...) = CELEBA(; split, kws...)
#CELEBA(Tx::Type; kws...) = CELEBA(; Tx, kws...)

getcelebafilename(i::Integer) = lpad(i, 6, "0") * ".jpg"
#readdir("/Users/matthewscott/.julia/datadeps/CELEBA/img_align_celeba")

function CELEBA(split::Symbol)
    DEPNAME = "CELEBA"
    features = FunctionalFileDataset(joinpath(datadep"CELEBA", "img_align_celeba"), getcelebafilename)
    CELEBA(split, features)
end

function CELEBA(Tx::Type, split::Symbol; dir=nothing, split_ratio=0.8)
    DEPNAME = "CELEBA"

    @assert split ∈ (:train, :test)

    main_dataset = FunctionalFileDataset(datadir("CELEBA", "img_align_celeba"), getcelebafilename)
    if split == :train
        features = FunctionalSubDataset(main_dataset, :train, split_ratio)
    else
        features = FunctionalSubDataset(main_dataset, :test, split_ratio)
    end

    return CELEBA(split, features)
end

convert2image(::Type{<:CELEBA}, x::AbstractArray{<:Integer}) =
    convert2image(CELEBA, reinterpret(N0f8, convert(Array{UInt8}, x)))

function convert2image(::Type{<:CELEBA}, x::AbstractArray{T,N}) where {T,N}
    @assert N == 3 || N == 4
    x = permutedims(x, (3, 2, 1, 4:N...))
    ImageCore = ImageShow.ImageCore
    return ImageCore.colorview(ImageCore.RGB, x)
end


# DEPRECATED INTERFACE, REMOVE IN v0.7 (or 0.6.x)
function Base.getproperty(::Type{CELEBA}, s::Symbol)
    if s == :traintensor
        @warn "CIFAR10.traintensor() is deprecated, use `CIFAR10(split=:train).features` instead."
        traintensor(T::Type=N0f8; kws...) = traintensor(T, :; kws...)
        traintensor(i; kws...) = traintensor(N0f8, i; kws...)
        function traintensor(T::Type, i; dir=nothing)
            CELEBA(; split=:train, Tx=T, dir)[i][1]
        end
        return traintensor
    elseif s == :testtensor
        @warn "CIFAR10.testtensor() is deprecated, use `CIFAR10(split=:test).features` instead."
        testtensor(T::Type=N0f8; kws...) = testtensor(T, :; kws...)
        testtensor(i; kws...) = testtensor(N0f8, i; kws...)
        function testtensor(T::Type, i; dir=nothing)
            CELEBA(; split=:test, Tx=T, dir)[i][1]
        end
        return testtensor
    elseif s == :trainlabels
        @warn "CIFAR10.trainlabels() is deprecated, use `CIFAR10(split=:train).targets` instead."
        trainlabels(; kws...) = trainlabels(:; kws...)
        function trainlabels(i; dir=nothing)
            CELEBA(; split=:train, dir)[i][2]
        end
        return trainlabels
    elseif s == :testlabels
        @warn "CIFAR10.testlabels() is deprecated, use `CIFAR10(split=:test).targets` instead."
        testlabels(; kws...) = testlabels(:; kws...)
        function testlabels(i; dir=nothing)
            CELEBA(; split=:test, dir)[i][2]
        end
        return testlabels
    elseif s == :traindata
        @warn "CIFAR10.traindata() is deprecated, use `CIFAR10(split=:train)[:]` instead."
        traindata(T::Type=N0f8; kws...) = traindata(T, :; kws...)
        traindata(i; kws...) = traindata(N0f8, i; kws...)
        function traindata(T::Type, i; dir=nothing)
            CELEBA(; split=:train, Tx=T, dir)[i]
        end
        return traindata
    elseif s == :testdata
        @warn "CIFAR10.testdata() is deprecated, use `CIFAR10(split=:test)[:]` instead."
        testdata(T::Type=N0f8; kws...) = testdata(T, :; kws...)
        testdata(i; kws...) = testdata(N0f8, i; kws...)
        function testdata(T::Type, i; dir=nothing)
            CELEBA(; split=:test, Tx=T, dir)[i]
        end
        return testdata
    elseif s == :convert2image
        @warn "CIFAR10.convert2image(x) is deprecated, use `convert2image(CIFAR10, x)` instead"
        return x -> convert2image(CELEBA, x)
    else
        return getfield(CIFAR10, s)
    end
end
