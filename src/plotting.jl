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
