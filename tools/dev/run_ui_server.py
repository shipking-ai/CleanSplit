"""Dev: run the CleanSplit UI server against an existing outputs folder (no desktop window)."""
import sys

from cleansplit.ui.server import serve

serve(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8770, out_root=sys.argv[2] if len(sys.argv) > 2 else "outputs")
