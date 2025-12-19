import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    def test_specparam2pandas_dimensions():
        """Test that specparam2pandas output has correct dimensions."""
        from spectral.specparam import specparam2pandas
        from specparam import SpectralGroupModel
        from specparam.sim import sim_group_power_spectra

        # Create test data
        n_models = 5

        freqs, powers = sim_group_power_spectra(
            n_models, [1, 50], {"fixed": [0, 1]}, {"gaussian": [10, 0.25, 2]}
        )

        # Fit model
        fg = SpectralGroupModel(verbose=False)
        fg.fit(freqs, powers)

        # Convert to dataframe
        df = specparam2pandas(fg)

        # Test dimensions
        assert len(df) >= n_models, f"Expected at least {n_models} rows, got {len(df)}"

        # Test columns
        expected_cols = [
            "ID",
            "offset",
            "exponent",
            "error_mae",
            "gof_rsquared",
            "CF",
            "PW",
            "BW",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        print(f"✓ Test passed: {len(df)} rows, {len(df.columns)} columns")
        return df  # Return for inspection in marimo

    return (test_specparam2pandas_dimensions,)


@app.cell
def _(test_specparam2pandas_dimensions):
    df = test_specparam2pandas_dimensions()
    df
    return


if __name__ == "__main__":
    app.run()
