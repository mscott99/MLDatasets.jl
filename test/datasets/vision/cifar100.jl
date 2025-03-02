n_features = (32, 32, 3)
n_targets = (coarse=1, fine=1)

@testset "trainset" begin
    d = CIFAR100()

    @test d.split == :train
    @test extrema(d.features) == (0, 1)
    @test convert2image(d, 1) isa AbstractMatrix{<:RGB}
    @test convert2image(d, 1:2) isa AbstractArray{<:RGB,3}

    test_supervised_array_dataset(d;
        n_features, n_targets, n_obs=50000,
        Tx=Float32, Ty=Int)

    d = CIFAR100(:train)
    @test d.split == :train 
    d = CIFAR100(Int, :train)
    @test d.split == :train 
    @test d.features isa AbstractArray{Int}
end

@testset "testset" begin 
    d = CIFAR100(split=:test, Tx=UInt8)

    @test d.split == :test
    @test extrema(d.features) == (0, 255)
    @test convert2image(d, 1) isa AbstractMatrix{<:RGB}
    @test convert2image(d, 1:2) isa AbstractArray{<:RGB,3}

    test_supervised_array_dataset(d;
        n_features, n_targets, n_obs=10000,
        Tx=UInt8, Ty=Int)
end
