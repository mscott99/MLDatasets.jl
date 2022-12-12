function setup_functional_filedataset_test()
    files = [
        "root/f1.csv",
        "root/f2.csv",
        "root/f3.csv",
        "root/f4.csv"
    ]

    for (i, file) in enumerate(files)
        paths = splitpath(file)[1:(end-1)]
        root = ""
        for p in paths
            fullp = joinpath(root, p)
            isdir(fullp) || mkdir(fullp)
            root = fullp
        end

        open(file, "w") do io
            write(io, "a,b,c\n")
            write(io, join(i .* [1, 2, 3], ","))
        end
    end
    return files
end
getfilename(index::Integer) = "f$(index).csv"

cleanup_filedataset_test() = rm("root"; recursive=true)

@testset "FunctionalFileDataset" begin
    #using MLDatasets: FunctionalFileDataset, FunctionalSubDataset
    files = setup_functional_filedataset_test()
    dataset = FunctionalFileDataset(f -> CSV.read(f, DataFrame), "root", getfilename)
    @test numobs(dataset) == length(files)
    for (i, file) in enumerate(files)
        true_obs = CSV.read(file, DataFrame)
        @test getobs(dataset, i) == true_obs
    end

    subtestdataset = FunctionalSubDataset(dataset, :test, 3 / 4)
    @test numobs(subtestdataset) == 1
    @test getobs(subtestdataset, 1) == CSV.read(files[4], DataFrame)
    @test_throws DataType getobs(subtestdataset, 2)
    @test_throws DataType getobs(subtestdataset, 0)
    cleanup_filedataset_test()
end