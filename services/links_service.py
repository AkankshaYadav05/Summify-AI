from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
	parsed = urlparse(url)
	hostname = (parsed.hostname or "").lower()

	if hostname in ("youtu.be", "www.youtu.be"):
		return parsed.path.lstrip("/")

	if "youtube" in hostname:
		params = parse_qs(parsed.query)
		if "v" in params and params["v"]:
			return params["v"][0]

	raise ValueError("Invalid YouTube URL.")


def get_youtube_transcript(url: str) -> str:
	video_id = extract_video_id(url)
	from youtube_transcript_api import YouTubeTranscriptApi

	transcript = None
	if hasattr(YouTubeTranscriptApi, "get_transcript"):
		transcript = YouTubeTranscriptApi.get_transcript(video_id)
	else:
		ytt = YouTubeTranscriptApi()
		if hasattr(ytt, "fetch"):
			transcript = ytt.fetch(video_id)
		elif hasattr(ytt, "get_transcript"):
			transcript = ytt.get_transcript(video_id)
		else:
			raise RuntimeError("Unsupported youtube_transcript_api version")

	parts = []
	for snippet in transcript:
		if isinstance(snippet, dict):
			parts.append(snippet.get("text", ""))
		else:
			parts.append(getattr(snippet, "text", ""))

	return " ".join(p for p in parts if p)


__all__ = ["extract_video_id", "get_youtube_transcript"]

