using ConsensusFitting
using Random
using Test

@testset "RANSAC line fitting" begin
    Random.seed!(42)

    # True model: y = a_true * x + b_true
    a_true, b_true = 2.0, 3.0
    n_inliers  = 100
    n_outliers = 40

    # Inlier points: near the true line with small Gaussian noise
    x_in = collect(range(-10.0, 10.0; length=n_inliers))
    y_in = a_true .* x_in .+ b_true .+ 0.2 .* randn(n_inliers)

    # Outlier points: uniformly scattered over a wider region
    x_out = -10.0 .+ 20.0 .* rand(n_outliers)
    y_out = -25.0 .+ 50.0 .* rand(n_outliers)

    # Pack into a 2 × N matrix (each column is one [x; y] point)
    data = [vcat(x_in, x_out)'; vcat(y_in, y_out)']

    # fittingfn: fit y = a*x + b through exactly 2 points
    function fit_line(pts)
        x1, y1 = pts[1, 1], pts[2, 1]
        x2, y2 = pts[1, 2], pts[2, 2]
        isapprox(x1, x2; atol=1e-10) && return []  # vertical → degenerate
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return [a, b]
    end

    # distfn: classify using vertical (y) residual
    function line_dist(M, x, t)
        a, b = M[1], M[2]
        residuals = abs.(x[2, :] .- (a .* x[1, :] .+ b))
        inliers = findall(residuals .< t)
        return inliers, M
    end

    M, inliers = ransac(data, fit_line, line_dist, 2, 0.5)

    # Recovered slope and intercept should be close to the true values
    @test abs(M[1] - a_true) < 0.1
    @test abs(M[2] - b_true) < 0.1

    # Should recover the majority of the inlier points
    @test length(inliers) ≥ round(Int, 0.9 * n_inliers)

    # All returned indices must be valid column indices of data
    @test all(1 .≤ inliers .≤ size(data, 2))

    # Error on too few points
    baddata = rand(2, 1)  # only 1 point, need s=2
    @test_throws ErrorException ransac(baddata, identity, (M, x, t) -> ([], M), 2, 0.5)
end
