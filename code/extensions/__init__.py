"""Beyond-the-paper experiments for PatchTST.

Two experiments not in Nie et al. 2023:

  eval_extensions.patch_shuffle_curve(...)
        Inference-only. Permute a random fraction of patches before they
        enter the encoder, measure MSE degradation. Tests how strongly the
        model relies on temporal patch ordering.

  channel_grouping.ChannelGroupedPatchTST + cluster_channels_by_correlation(...)
        Requires training. Splits the M channels into k clusters; within each
        cluster channels are mixed inside a token, across clusters they're
        independent (sharing weights). k=1 reduces to channel-mixing,
        k=M to channel-independence. Sweeping k explores the spectrum the
        paper presents as binary.
"""
