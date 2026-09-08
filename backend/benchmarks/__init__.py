"""Image-quality benchmark harness.

Isolated from the production runtime on purpose: nothing in `app/` imports this
package, and this package changes nothing it measures. It reads result files
and reports numbers, so a future model or post-process can be argued about with
measurements rather than impressions.
"""
