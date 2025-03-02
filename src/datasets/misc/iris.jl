function __init__iris()
    DEPNAME = "Iris"
    LINK = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
    DOCS = "https://archive.ics.uci.edu/ml/datasets/Iris"
    DATA = "iris.data"

    register(DataDep(
        DEPNAME,
        """
        Dataset: The Iris dataset
        Website: $DOCS
        """,
        LINK .* [DATA],
        "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"  # if checksum omitted, will be generated by DataDeps
    ))
end

"""
    Iris(; as_df = true, dir = nothing)

Fisher's classic iris dataset. 

Measurements from 3 different species of iris: setosa, versicolor and
virginica. There are 50 examples of each species.

There are 4 measurements for each example: sepal length, sepal width, petal
length and petal width.  The measurements are in centimeters.

The module retrieves the data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

NOTE: no pre-defined train-test split for this dataset. 

# Arguments

$ARGUMENTS_SUPERVISED_TABLE

# Fields

$FIELDS_SUPERVISED_TABLE

# Methods

$METHODS_SUPERVISED_TABLE

# Examples

```julia-repl
julia> dataset = Iris()
Iris:
  metadata => Dict{String, Any} with 4 entries
  features => 150×4 DataFrame
  targets => 150×1 DataFrame
  dataframe => 150×5 DataFrame


julia> dataset[1:2]
(2×4 DataFrame
 Row │ sepallength  sepalwidth  petallength  petalwidth 
     │ Float64      Float64     Float64      Float64    
─────┼──────────────────────────────────────────────────
   1 │         5.1         3.5          1.4         0.2
   2 │         4.9         3.0          1.4         0.2, 2×1 DataFrame
 Row │ class       
     │ String15    
─────┼─────────────
   1 │ Iris-setosa
   2 │ Iris-setosa)

julia> X, y = Iris(as_df=false)[:]
([5.1 4.9 … 6.2 5.9; 3.5 3.0 … 3.4 3.0; 1.4 1.4 … 5.4 5.1; 0.2 0.2 … 2.3 1.8], InlineStrings.String15["Iris-setosa" "Iris-setosa" … "Iris-virginica" "Iris-virginica"])
```
"""
struct Iris <: SupervisedDataset
    metadata::Dict{String, Any}
    features
    targets
    dataframe
end

function Iris(; dir = nothing, as_df = true)
    path = datafile("Iris", "iris.data", dir)
    df = read_csv(path, header=0)
    DataFrames.rename!(df, ["sepallength", "sepalwidth", "petallength", "petalwidth", "class"])

    features = df[!, DataFrames.Not(:class)]
    targets = df[!, [:class]]

    metadata = Dict{String, Any}()
    metadata["path"] = path
    metadata["n_observations"] = size(df, 1)
    metadata["feature_names"] = names(features)
    metadata["target_names"] = names(targets)

    if !as_df
        features = df_to_matrix(features)
        targets = df_to_matrix(targets)
        df = nothing
    end

    return Iris(metadata, features, targets, df)
end

# deprecated in v0.6
function Base.getproperty(::Type{Iris}, s::Symbol)
    if s == :features
        @warn "Iris.features() is deprecated, use `Iris().features` instead."
        return () -> Iris(as_df=false).features
    elseif s == :labels
        @warn "Iris.labels() is deprecated, use `Iris().targets` instead."
        return () -> Iris(as_df=false).targets |> vec
    else 
        return getfield(Iris, s)
    end
end
