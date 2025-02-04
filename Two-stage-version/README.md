__4/2/2025__<br />
__Overview:__<br />
This version was created after the single stage version as an attempt to lower the false positive rate and still maintain somewhat acceptable recall. It therefore trains and applies two models. First one is designed to have a high recall rate at the expense of precision (this is done mainly through higher minority class weight and lower threshold) and the secodn model refines the initial selection with forcus on precision.

In the end this ends up with very high precision (low false positive rate) and a in compatrison with the previous version worse but still accteptable recall rate of about 70%. This in the end results in a bigger area under the precision-recall curve. Based purely on this metric this version may seem better but that might not be the case as anyone using this type of model may prefer the higher recall rate at the expense of more false positives simply from the standpoint of wanting to stop as much fraud as possible.

__Development:__<br />
Not much chainged, this was a short additional detour from the project, the challenges were the same as with the previous model, just in a shorter time frame.
