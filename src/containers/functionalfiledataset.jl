"""
    FunctionalFileDataset([loadfn = FileIO.load,] folderpath, makefilenamefunction
"""
struct FunctionalFileDataset <: AbstractDataContainer
    loadfn::Function
    folderpath::String
    makefilename::Function
    thelength::Int
end

function FunctionalFileDataset(folderpath, makefilename)
    thelength = Base.length(readdir(folderpath))
    FunctionalFileDataset(FileIO.load, folderpath, makefilename, thelength)
end

FunctionalFileDataset(folderpath, makefilename) = FunctionalFileDataset(FileIO.load, folderpath, makefilename)

Base.getindex(dataset::FunctionalFileDataset, i::Integer) = dataset.loadfn(joinpath(dataset.folderpath, dataset.makefilename(i)))[:, :, :, 1]
Base.getindex(dataset::FunctionalFileDataset, is::AbstractVector) = map(Base.Fix1(getobs, dataset), is)
Base.length(dataset::FunctionalFileDataset) = dataset.thelength

"""
FileDataset([loadfn = FileIO.load,] folderpath, makefilenamefunction)
"""
struct FunctionalSubDataset{L<:FunctionalFileDataset} <: AbstractDataContainer
    dataset::L
    startindex::Int
    endindex::Int
    function FunctionalSubDataset(dataset::L, startindex::Int, endindex::Int) where {L<:FunctionalFileDataset}
        if endindex > length(dataset)
            throw("End index is greater than length of dataset")
        end
        if startindex < 1
            throw("Start index is less than 1")
        end
        if startindex > length(dataset)
            throw("Start index is greater than length of dataset")
        end
        if startindex > endindex
            throw("Start index is greater than end index")
        end
        new{typeof(dataset)}(dataset, startindex, endindex)
    end
end

function FunctionalSubDataset(dataset::FunctionalFileDataset, datasplit, splitratio)
    midpoint = round(Int, length(dataset) * splitratio)
    if datasplit == :test
        return FunctionalSubDataset(dataset, midpoint + 1, length(dataset))
    elseif datasplit == :train
        return FunctionalSubDataset(dataset, 1, midpoint)
    else
        throw("Invalid datasplit")
    end
end

Base.length(dataset::FunctionalSubDataset) = dataset.endindex - dataset.startindex + 1
Base.getindex(dataset::FunctionalSubDataset, i::Integer) = (i < 1 || i > length(dataset)) ? throw(BoundsError) : Base.getindex(dataset.dataset, i + dataset.startindex - 1)
Base.getindex(dataset::FunctionalSubDataset, is::AbstractVector) = Base.getindex(dataset.dataset, is .+ dataset.startindex .- 1)