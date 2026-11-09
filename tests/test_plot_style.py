"""Export contracts: exact panels, transparent legends, and shared names."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/erpm_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/erpm-cache")
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from scripts import plot_style as style


class PlotExportTests(unittest.TestCase):
    def setUp(self):
        style.apply_style()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_legend_and_exact_clean_panels(self):
        for name in ("calcium", "flux"):
            fig, ax = style.figure()
            for key, (label, color, linestyle) in style.CONDITIONS.items():
                ax.plot([0, 1], [0, 1], label=label, color=color, linestyle=linestyle)
            style.save(fig, self.root / f"{name}.png", legend_name="legend.png")
            with Image.open(self.root / "fig" / f"{name}.png") as clean:
                self.assertEqual(clean.size, (900, 900))
            with Image.open(self.root / f"{name}.png") as view:
                self.assertGreater(view.height, 900)
        self.assertEqual(len(list((self.root / "fig").glob('*legend.png'))), 1)
        with Image.open(self.root / "fig/legend.png") as legend:
            self.assertEqual(legend.mode, "RGBA")
            self.assertEqual(legend.getextrema()[3], (0, 255))

    def test_single_series_needs_no_legend(self):
        fig, ax = style.figure()
        ax.plot([0, 1], [1, 2], label="Ca")
        style.save(fig, self.root / "trace.pdf")
        self.assertEqual((self.root / "trace.png").read_bytes(),
                         (self.root / "fig/trace.png").read_bytes())
        self.assertFalse(list(self.root.rglob('*.pdf')))
        self.assertFalse(list(self.root.rglob('*legend.png')))

    def test_export_hides_categorical_legend_but_preserves_colorbar(self):
        fig, ax = style.figure()
        ax.plot([0, 1], [0, 1], label="first")
        ax.plot([0, 1], [1, 0], label="second")
        ax.legend()
        im = ax.imshow([[0, 1], [1, 0]], cmap=style.blue_colormap())
        fig.colorbar(im, ax=ax)
        calls = []
        original = fig.savefig
        def record(path, **kwargs):
            calls.append((len(fig.axes), len(fig.legends), ax.get_legend()))
            return original(path, **kwargs)
        fig.savefig = record
        style.save(fig, self.root / 'heat.png')
        self.assertEqual(calls, [(2, 0, None), (2, 1, None)])


if __name__ == '__main__':
    unittest.main()
