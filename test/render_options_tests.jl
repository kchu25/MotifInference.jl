# CPU-only: the outside-window mutation option reaches every rendering entry point.
using MotifInference
using Test

@testset "show_outside_mutations keyword" begin
    # `hasmethod(f, argtypes, kwnames)` (public API) is true when the method that `argtypes`
    # dispatches to declares the keywords, or takes `kwargs...`. A made-up keyword tells the
    # two apart: only a `kwargs...` method accepts it.
    kw = (:show_outside_mutations,)
    bogus = (:no_such_keyword_for_this_test,)
    accepts(f, T) = hasmethod(f, T, kw)
    declares(f, T) = accepts(f, T) && !hasmethod(f, T, bogus)

    TRC = MotifInference.training_and_rendering_config
    # run_method(trc; ...) and run_method(trc, data; ...) declare it.
    @test declares(MotifInference.run_method, Tuple{TRC})
    @test declares(MotifInference.run_method, Tuple{TRC, Any})
    # The NamedTuple / path / in-memory methods forward it through `kwargs...`.
    @test accepts(MotifInference.run_method, Tuple{NamedTuple})
    @test accepts(MotifInference.run_method, Tuple{String})
    @test accepts(MotifInference.run_method, Tuple{Vector{String}, Vector{Float64}})
    @test declares(MotifInference._run_one!, Tuple{Any, Any})
    @test declares(MotifInference.render_html, NTuple{10, Any})
    # The glyph root it is passed to (mut case).
    @test declares(MotifInference.GlyphEctoplasm.plot_motifs_mut_case, NTuple{4, Any})
end
