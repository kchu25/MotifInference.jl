
const motif_names = ["pairs", "triplets", "quadruplets", "quintuplets"]

# Collect extrema across all DataFrames without allocation
function obtain_xlim(contributions_df_filtered_singletons, dfs)
    xlim = mapreduce(extrema, 
        (a, b) -> (min(a[1], b[1]), max(a[2], b[2])),
        (contributions_df_filtered_singletons.banzhaf, 
            (df.banzhaf for df in dfs)...))

    return xlim
end

# function plot_motifs_conv_case(data, m, motif_sizes, 
#         contributions_df_filtered_singletons, dfs, pts;
#         interaction_summary=nothing,
#         nav_page_count=4,
#         enable_colored_borders = true,
#         use_unified=true,
#         dpi=65, 
#         save_path="tmp", 
#         page_title="n/a", 
#         rna=false
#         );

#     # motif rendering
#     xlim = obtain_xlim(contributions_df_filtered_singletons, dfs)

#     config = ConvMotifConfig(data; 
#         filter_len=m.hp.pfm_len, dpi=dpi, save_path=save_path, xlim=xlim)
        
#     json_motifs = init_json_dict()
#     html_dict = init_dict_for_html_render()

#     next_idx = process_singletons!(
#         contributions_df_filtered_singletons, config, json_motifs, html_dict; start_idx=1, rna=rna)

#     group_ids = [motif_names[min(size-1, 4)] for size in motif_sizes]
#     button_texts = ["$(size)-motifs" for size in motif_sizes]

#     for (motif_size, group_id, button_text) in zip(motif_sizes, group_ids, button_texts)
#         @info "Processing multi-motifs of size: $(motif_size)"
#         @time next_idx = process_multi_motifs!(dfs, 
#             config, json_motifs, html_dict;
#                 interaction_summary=interaction_summary,
#                 motif_size=motif_size, group_id=group_id, 
#                 button_text=button_text, start_idx=next_idx, rna=rna                
#                 )
#     end

#     # Generate combined panel figure 
#     data_pairs = [
#         (pts.predictions, pts.labels, "Predictions", "Labels", "Predictions vs Labels"),
#         (pts.proc_prod, pts.labels, "Learned Predictions", "Labels", "Learned Predictions vs Labels"),
#         (pts.proc_prod, pts.predictions, "Learned Predictions", "Predictions", "Learned Predictions vs Predictions")
#     ]
#     GlyphEctoplasm.publication_scatter_panel(data_pairs, save_path=joinpath(save_path, "generalization.png"))


#     render_and_save_outputs!(json_motifs, html_dict, 1; 
#         html_template=html_template_unified, 
#         script_template=script_template,
#         css_template=template_css,
#         nav_page_count=nav_page_count,
#         sequence_paths=[""],
#         page_title=page_title,
#         save_path=save_path, 
#         enable_colored_borders = enable_colored_borders,
#         use_unified=use_unified
#         )
# end

function plot_motifs_mut_case(data, m, save_path,
    contributions_df_filtered_singletons, dfs;
    use_rna=false,
    off_region_search=true, # figure this out later
    float_type=Float32,
    dpi=65,
    reduction_on_ref=false,
    singleton_filter_pareto_rank=1, # figure out
    split_by_sign=true, # figure out
    sort_globally=true, # figure out
    sort_by_pareto=true, # figure out
    nav_page_count=4, # figure out
    page_title = "Mutation Regions Analysis",
    use_unified = true,
    enable_colored_borders = true
    )
    xlim = obtain_xlim(contributions_df_filtered_singletons, dfs)

    # Single configuration object for all mutation region analysis
    m_config = MutationRegionConfig(data;
        filter_len = m.receptive_field,
        float_type = float_type,
        use_rna = use_rna,
        off_region_search = off_region_search,
        xlim = xlim,
        save_path = save_path,
        dpi = dpi,
        reduction_on_ref = reduction_on_ref
    )

    all_metadata = prepare_and_collect_mutation_metadata(
        contributions_df_filtered_singletons, dfs, data, m_config;
        singleton_filter_pareto_rank = singleton_filter_pareto_rank,
        split_by_sign = split_by_sign  # Splits by sign when computing Pareto ranks for singletons
    );

    json_motifs = init_json_dict()
    html_dict = init_dict_for_html_render()

    register_mutation_region_motifs!(
        json_motifs, html_dict, all_metadata;
        start_idx = 1,
        sort_globally = sort_globally,    # Enable hierarchical sorting
        sort_by_pareto = sort_by_pareto    # Use Pareto ranking within groups
    )

    render_and_save_outputs!(json_motifs, html_dict, 1;
        html_template = html_template_unified,
        script_template = script_template,
        css_template = template_css,
        save_path = save_path,
        nav_page_count = nav_page_count,  # Show navigation for 4 pages: Pattern influence, Generalization, Readme, Statistics
        sequence_paths = [""],
        page_title = page_title,
        use_unified = use_unified,
        enable_colored_borders = enable_colored_borders
    )
end