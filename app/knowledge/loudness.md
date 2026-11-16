# Loudness and peaks

## Loudness targets and streaming normalisation
tags: loudness, lufs, streaming, normalisation, spotify, apple music, youtube, master too loud, master too quiet

Streaming services normalise playback: Spotify and YouTube aim at roughly -14 LUFS integrated, Apple Music around -16 LUFS. A master at -7 LUFS is simply turned down by about 7 dB, so beyond a point, extra loudness only buys you lost transients and distortion. Pick a loudness for the genre and the feel you want, not to "win" on a playlist. As a rough guide: acoustic and jazz -18 to -14 LUFS, pop and rock -10 to -8, EDM and trap -8 to -6. Always compare against references at matched loudness (turn the louder one down) or you will judge "louder" as "better".

## Raising loudness without crushing the mix
tags: loudness, quiet master, limiter, clipper, gain staging, too quiet, lufs low

If the master is quieter than the reference, do not just push one limiter harder. Spread the gain over stages: gentle bus compression (1.5:1 to 2:1, 1-2 dB of gain reduction), a soft clipper taking 1-2 dB off the fastest transients, then a final limiter doing another 2-3 dB. Each stage working lightly sounds cleaner than one stage working hard. Loudness is also a mix problem: an uncontrolled sub, or resonances around 2-5 kHz, eat headroom. High-pass elements that don't need sub energy and tame harsh peaks before mastering.

## Master too loud or over-limited
tags: too loud, over-compressed, limiter, plr low, crest factor low, squashed, distortion, pumping

Signs of an over-limited master: a peak-to-loudness ratio (PLR) below about 7 dB, a crest factor under 6 dB, audible pumping, kick and snare that lose their attack, and distortion on sustained bass notes. Back the limiter off until gain reduction peaks at 3-4 dB, use a slower release (100-300 ms) to avoid distorting the low end, and if you need the loudness, get it upstream with clipping on drums or parallel compression. Check the result at matched loudness against the louder version: often the quieter one sounds bigger.

## True peak and inter-sample clipping
tags: true peak, dbtp, inter-sample, clipping, ceiling, lossy, mp3, aac, encoding

Sample peaks can read below 0 dBFS while the reconstructed analogue waveform goes above it: these are inter-sample peaks, and they distort in DACs and, above all, in lossy encoders (MP3, AAC, Ogg) used by streaming services. Set the limiter's ceiling to -1.0 dBTP with true-peak detection (oversampling) enabled; for very loud masters (above -8 LUFS) -1.5 or -2 dBTP is safer, since encoding adds overshoot. Spotify recommends -1 dBTP, or -2 dBTP for masters louder than -14 LUFS.

## Digital clipping in the file
tags: clipping, clipped samples, distortion, 0 dbfs, overs, gain staging

Samples stuck at full scale mean the file was clipped when it was rendered or printed. Find out where: check the master bus meter, any plug-in on the master that adds gain after the limiter, and whether the DAW renders in fixed point. Render at 24-bit or 32-bit float, keep the limiter last in the chain, and leave the ceiling below 0 dBFS. If the clipping is intentional (a clipper used for colour), the waveform should still not sit at exactly 0 dBFS in the delivered file.

## Loudness range and macro-dynamics
tags: lra, loudness range, dynamics, macro dynamics, sections, verse, chorus, automation

Loudness range (LRA, in LU) measures how much loudness varies between sections. Dense electronic and pop masters often sit around 3-6 LU; rock 5-8; acoustic, film and classical 8-15 or more. An LRA that is very low makes a track feel flat: chorus and verse hit the same wall. Bring contrast back with arrangement (fewer elements in the verse), level automation of whole sections (+1 to +2 dB into the chorus), and less master-bus compression. An LRA much higher than the reference usually means quiet intros or breakdowns; that is fine artistically, but those sections may feel weak on a playlist.
