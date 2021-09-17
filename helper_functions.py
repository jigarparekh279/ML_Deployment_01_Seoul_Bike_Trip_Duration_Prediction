def pca_pipe(data, n):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    pca = PCA(n)

    pipe = Pipeline([('scalar',StandardScaler()), ('pca',pca)])
    PCs = pipe.fit_transform(data)

    return PCs, pca.explained_variance_ratio_

