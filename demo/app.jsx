// Serif display, navy accent, centered hero.
const PyramidForcingApp = () => {
  const navy = 'oklch(0.42 0.09 250)';
  const navyDeep = 'oklch(0.32 0.10 250)';
  const ink = 'oklch(0.18 0.01 250)';
  const muted = 'oklch(0.50 0.01 250)';
  const rule = 'oklch(0.90 0.01 250)';
  const bg = '#ffffff';
  const tint = 'oklch(0.97 0.012 250)';

  // Source Serif 4 is a variable font with an optical-sizing (opsz) axis,
  // so the same family stays elegant at the title and crisp at body size
  // — no need for separate display vs text faces.
  const display = `'Source Serif 4', Georgia, serif`;
  const serif = `'Source Serif 4', Georgia, serif`;
  const sans = `'Inter', system-ui, sans-serif`;
  const mono = `'JetBrains Mono', ui-monospace, monospace`;

  const Btn = ({ icon, label, href = '#', bg }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" style={{
      display: 'inline-flex', alignItems: 'center', gap: 10,
      padding: '11px 20px', borderRadius: 999,
      background: bg || ink, color: '#fff',
      fontFamily: sans, fontSize: 14, fontWeight: 500,
      textDecoration: 'none', letterSpacing: 0.1,
    }}>
      <span style={{ display: 'inline-flex', width: 20, height: 20, alignItems: 'center', justifyContent: 'center' }}>{icon}</span>
      <span>{label}</span>
    </a>
  );

  // arXiv: official X mark (path from simple-icons), white on red bg
  const arxivIcon = (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="#fff" aria-hidden="true">
      <path d="M3.8423 0a1.0037 1.0037 0 0 0-.922.6078c-.1536.3687-.0438.6275.2938 1.1113l6.9185 8.3597-1.0223 1.1058a1.0393 1.0393 0 0 0 .003 1.4229l1.2292 1.3135-5.4391 6.4444c-.2803.299-.4538.823-.2971 1.1986a1.0253 1.0253 0 0 0 .9585.635.9133.9133 0 0 0 .6891-.3405l5.783-6.126 7.4902 8.0051a.8527.8527 0 0 0 .6835.2597.9575.9575 0 0 0 .8777-.6138c.1577-.377-.017-.7502-.306-1.1407l-7.0518-8.3418 1.0632-1.13a.9626.9626 0 0 0 .0089-1.3165L4.6336.4639s-.3733-.4535-.768-.463zm0 .272h.0166c.2179.0052.4874.2715.5644.3639l.005.006.0052.0055 10.169 10.9905a.6915.6915 0 0 1-.0072.945l-1.0666 1.133-1.4982-1.7724-8.5994-10.39c-.3286-.472-.352-.6183-.2592-.841a.7307.7307 0 0 1 .6704-.4401Zm14.341 1.5701a.877.877 0 0 0-.6554.2418l-5.6962 6.1584 1.6944 1.8319 5.3089-6.5138c.3251-.4335.479-.6603.3247-1.0292a1.1205 1.1205 0 0 0-.9763-.689zm-7.6557 12.2823 1.3186 1.4135-5.7864 6.1295a.6494.6494 0 0 1-.4959.26.7516.7516 0 0 1-.706-.4669c-.1119-.2682.0359-.6864.2442-.9083l.0051-.0055.0047-.0055z" />
    </svg>
  );
  // GitHub: inline SVG path (FontAwesome 'fab fa-github' would also work)
  const githubIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff" aria-hidden="true">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.57.1.78-.25.78-.55v-2.05c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.95.1-.74.4-1.25.73-1.54-2.55-.29-5.23-1.27-5.23-5.66 0-1.25.45-2.27 1.18-3.07-.12-.29-.51-1.46.11-3.04 0 0 .96-.31 3.15 1.17a10.94 10.94 0 0 1 5.74 0c2.18-1.48 3.14-1.17 3.14-1.17.62 1.58.23 2.75.11 3.04.74.8 1.18 1.82 1.18 3.07 0 4.4-2.69 5.36-5.25 5.65.41.36.78 1.06.78 2.14v3.17c0 .31.21.66.79.55C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z" />
    </svg>
  );

  const chefPrompt = "A dynamic scene captured in the style of a vibrant food photography, showcasing a chef skillfully chopping onions in a bustling kitchen. The chef, a middle-aged man with a weathered face and determined expression, skillfully slices the onions with quick, practiced movements. He wears a white apron tied neatly around his waist and a chef's hat perched atop his head. The background is a well-equipped kitchen, with stainless steel appliances and countertops cluttered with various cooking tools and ingredients. Steam rises from a pot on the stove, and sunlight filters through the window, casting a warm glow. A medium shot with the chef at the center, capturing the intensity of his work.";

  // Videos are hosted as GitHub Release assets to keep the repo small.
  // To bump the asset set, upload to a new release tag and update this URL.
  const videoBase = 'https://github.com/IF-LAB-PKU/Pyramid-Forcing/releases/download/demo-videos-v1';

  // Asset paths — see static/README.md for the full file map.
  const media = {
    heroFig: './static/images/Fig_1.webp',
    pipeline: './static/images/method_overview.webp',
    analysis: './static/images/head_attention.webp',
    // Qualitative samples from static/prompts.csv — the trailing 3-digit
    // index in the filename matches the prompt's `index` column directly.
    movieGen: [
      {
        src: `${videoBase}/pf30s-video_000.mp4`,
        prompt: 'A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.',
      },
      {
        src: `${videoBase}/pf30s-video_021.mp4`,
        prompt: 'A vibrant illustration in a whimsical cartoon style depicting a flock of paper airplanes fluttering through a dense jungle. The airplanes, resembling small birds, weave gracefully around towering trees, their wings fluttering gently. The jungle is lush and vibrant, with a variety of exotic plants and colorful flowers. The airplanes seem to migrate through the forest, creating a mesmerizing aerial dance. The background is rich with detailed textures, including sunlight filtering through the canopy, casting dappled shadows on the ground. A dynamic overhead view capturing the mid-flight action of the airplanes.',
      },
      {
        src: `${videoBase}/pf30s-video_028.mp4`,
        prompt: 'A cyberpunk-style illustration depicting a lone robot navigating a neon-lit cityscape. The robot stands tall with sleek, metallic armor, adorned with blinking lights and wires. Its eyes, glowing with a deep blue hue, scan the surroundings with curiosity. The background features towering skyscrapers, holographic advertisements, and crowded streets filled with various cyborgs and humans. The air is thick with smoke and the hum of technology. A medium shot from a high-angle perspective, capturing both the robot and the bustling city environment.',
      },
      {
        src: `${videoBase}/pf30s-video_048.mp4`,
        prompt: chefPrompt,
      },
      {
        src: `${videoBase}/pf30s-video_110.mp4`,
        prompt: "A close-up shot of a young woman driving a car, lost in thought as she gazes ahead. Raindrops blur the view of a green forest through the car window. She wears a sleek raincoat and sunglasses, her expression contemplative. Her hands gently grip the steering wheel, and her fingers tap rhythmically against it. The interior of the car is dimly lit, with water droplets clinging to the windshield. The blurred green forest and rain create a sense of mystery and introspection. The photo has a cinematic quality, capturing the moment just before a decision is made. A close-up shot from inside the car, focusing on the driver.",
      },
      {
        src: `${videoBase}/pf30s-video_062.mp4`,
        prompt: "A close-up shot of someone carefully pouring milk into a cup, with the milk flowing smoothly and filling the cup with a milky white color. The person's hand is steady, guiding the milk into the cup with precision. The background is blurred, showing a subtle kitchen setting with hints of cabinets and countertops. The photo has a soft, natural lighting effect, emphasizing the smoothness and elegance of the pouring action.",
      },
      {
        src: `${videoBase}/pf30s-video_063.mp4`,
        prompt: 'A detailed oil painting in a romantic style, showcasing a young woman standing amidst a vibrant garden filled with blooming flowers. She wears a floral-patterned dress, her hair loosely tied with wildflowers adorning it. Her expression is one of serene joy, with a gentle smile on her lips. She is framed by a variety of colorful blooms, including roses, tulips, and daisies, which surround her in a natural, organic arrangement. The background features a soft, pastel-colored sky with fluffy clouds, and a gentle breeze rustling through the petals. A medium shot with a slightly tilted angle, capturing the essence of spring and renewal.',
      },
      {
        src: `${videoBase}/pf30s-video_064.mp4`,
        prompt: "A cinematic scene from a classic western movie, featuring a rugged man riding a powerful horse through the vast Gobi Desert at sunset. The man, dressed in a dusty cowboy hat and a worn leather jacket, reins tightly on the horse's neck as he gallops across the golden sands. The sun sets dramatically behind them, casting long shadows and warm hues across the landscape. The background is filled with rolling dunes and sparse, rocky outcrops, emphasizing the harsh beauty of the desert. A dynamic wide shot from a low angle, capturing both the man and the expansive desert vista.",
      },
      {
        src: `${videoBase}/pf30s-video_068.mp4`,
        prompt: "A charming illustration in a watercolor style of a young white rabbit wearing glasses and reading a newspaper. The rabbit has soft fur, large round ears, and gentle, curious eyes. It sits upright on a cozy armchair, one paw holding the newspaper and the other resting on its knee. The background features a warm living room with a fireplace, a few books on a side table, and a blurred view of a window with falling leaves. The rabbit's expression is one of focused interest, with a slight smile playing on its lips. A close-up shot from a slightly elevated angle, capturing the rabbit's detailed features and the newspaper's headlines.",
      },
    ],
    // Per-row baseline comparison. Each row shares one prompt; columns line up as
    // SF / RF / DF baselines on the left, the +PF (ours) variant on the right.
    // The 4th column in a row carries the highlight flag.
    compare30s: [
      {
        prompt: "A zoom-in shot focusing on the face of a young woman sitting on a bench in the middle of an empty school gym. The woman has long wavy brown hair cascading down her shoulders and soft, warm hazel eyes. She wears a simple white t-shirt and blue jeans, her hands resting gently on her knees. Her expression is serene, with a slight smile playing on her lips. The gymnasium is mostly empty, with only a few scattered bleachers and a basketball hoop in the background. The lighting is soft and natural, creating gentle shadows under her eyes and nose. The overall atmosphere is peaceful and contemplative.",
        cells: [
          { src: `${videoBase}/rf30s-video_092.mp4`, label: 'Rolling Forcing' },
          { src: `${videoBase}/sf30s-video_092.mp4`, label: 'Self Forcing' },
          { src: `${videoBase}/df30s-video_092.mp4`, label: 'SF + Deep Forcing' },
          { src: `${videoBase}/pf30s-video_092.mp4`, label: 'SF + Pyramid Forcing', highlight: true },
        ],
      },
      {
        prompt: 'A close-up shot of a steaming cappuccino in a ceramic cup, with a rich brown foam on top and a slight milk swirl pattern. The cup has a simple yet elegant design, with a white handle and a light brown body. The background is a cozy café with warm lighting, wooden tables, and a few patrons chatting in the corner. The cappuccino is freshly made, with a hint of steam rising from the surface, capturing the essence of a perfect morning beverage.',
        cells: [
          { src: `${videoBase}/rf30s-video_057.mp4`, label: 'Rolling Forcing' },
          { src: `${videoBase}/sf30s-video_057.mp4`, label: 'Self Forcing' },
          { src: `${videoBase}/cf30s-video_057.mp4`, label: 'Causal Forcing' },
          { src: `${videoBase}/cfpf30s-video_057.mp4`, label: 'CF + Pyramid Forcing', highlight: true },
        ],
      },
    ],
    compare60s: [
      {
        prompt: 'A macro shot focusing on the face of a young woman with freckles, her expression intense as she looks intently for something. Her freckles are scattered across her cheeks and nose, adding a playful charm to her face. Her eyes are wide and slightly squinted, peering closely at the object of her search. Her hair is loose, framing her face gently, with strands falling over her forehead. The background is blurred, but you can make out the faint outline of a table or desk where she is searching. The texture of her skin is smooth and 细腻，带有淡淡的红润。A close-up shot from a very close angle, capturing the natural and focused expression of the young woman.',
        cells: [
          { src: `${videoBase}/rf60s-video_095.mp4`, label: 'Rolling Forcing' },
          { src: `${videoBase}/sf60s-video_095.mp4`, label: 'Self Forcing' },
          { src: `${videoBase}/df60s-video_095.mp4`, label: 'SF + Deep Forcing' },
          { src: `${videoBase}/pf60s-video_095.mp4`, label: 'SF + Pyramid Forcing', highlight: true },
        ],
      },
      {
        // Row 058 is the CF-baseline row; the +PF column is the cf+pf augmented clip.
        prompt: 'A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement.',
        cells: [
          { src: `${videoBase}/rf60s-video_058.mp4`, label: 'Rolling Forcing' },
          { src: `${videoBase}/sf60s-video_058.mp4`, label: 'Self Forcing' },
          { src: `${videoBase}/cf60s-video_058.mp4`, label: 'Causal Forcing' },
          { src: `${videoBase}/cfpf60s-video_058.mp4`, label: 'CF + Pyramid Forcing', highlight: true },
        ],
      },
    ],
    // Per-component ablation: SF baseline → +Head Classify → +Ragged Cache → full PF.
    ablation: [
      {
        prompt: 'A vibrant illustration in a whimsical cartoon style depicting a flock of paper airplanes fluttering through a dense jungle. The airplanes, resembling small birds, weave gracefully around towering trees, their wings fluttering gently. The jungle is lush and vibrant, with a variety of exotic plants and colorful flowers. The airplanes seem to migrate through the forest, creating a mesmerizing aerial dance. The background is rich with detailed textures, including sunlight filtering through the canopy, casting dappled shadows on the ground. A dynamic overhead view capturing the mid-flight action of the airplanes.',
        cells: [
          { src: `${videoBase}/sf30s-video_021.mp4`, label: 'Self Forcing' },
          { src: `${videoBase}/Ablation-1HC30s-video_021.mp4`, label: '+ Head Classify' },
          { src: `${videoBase}/Ablation-2RC30s-video_021.mp4`, label: '+ Ragged Cache' },
          { src: `${videoBase}/pf30s-video_021.mp4`, label: 'Head + Pyramid KV' },
        ],
      },
    ],
  };

  // Renders a real <video> when src is provided, otherwise the placeholder
  // pattern so the page is still legible before media is dropped in.
  const VidCell = ({ label, prompt, src, w = '100%', h = 200, controls = true, autoPlay = false }) => (
    <div style={{ width: w }}>
      <div style={{
        width: '100%', height: h, background: tint,
        border: `1px solid ${rule}`, borderRadius: 6,
        position: 'relative', overflow: 'hidden',
        ...(src ? {} : {
          backgroundImage: `repeating-linear-gradient(135deg, transparent 0 14px, ${rule} 14px 15px)`,
        }),
      }}>
        {src ? (
          <video
            src={src}
            autoPlay={autoPlay}
            loop={autoPlay}
            muted={autoPlay}
            playsInline
            controls={controls}
            preload="metadata"
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          />
        ) : (
          <React.Fragment>
            <div style={{
              position: 'absolute', top: 10, left: 12,
              fontFamily: mono, fontSize: 10, color: muted, letterSpacing: 0.5,
              textTransform: 'uppercase', background: bg, padding: '3px 7px',
              border: `1px solid ${rule}`, borderRadius: 3,
            }}>{label}</div>
            <div style={{
              position: 'absolute', bottom: 10, left: 12, right: 12,
              fontFamily: mono, fontSize: 10, color: muted,
            }}>// {prompt}</div>
          </React.Fragment>
        )}
      </div>
    </div>
  );

  return (
    <div style={{
      width: 1100, boxSizing: 'border-box',
      fontFamily: serif, color: ink, background: bg,
      padding: '64px 80px 80px',
    }}>
      {/* Title */}
      <h1 style={{
        fontFamily: display, fontWeight: 600,
        fontSize: 56, lineHeight: 1.08, textAlign: 'center',
        margin: '0 0 6px', letterSpacing: -0.5, textWrap: 'balance',
      }}>
        Pyramid Forcing:
      </h1>
      <h2 style={{
        fontFamily: display, fontWeight: 500, fontStyle: 'italic',
        fontSize: 28, lineHeight: 1.2, textAlign: 'center',
        color: navyDeep, margin: '0 0 36px', textWrap: 'balance',
      }}>
        Head-Aware Pyramid KV Cache Policy for<br/>High-Quality Long Video Generation
      </h2>

      {/* Authors */}
      <div style={{
        textAlign: 'center', fontFamily: serif, fontSize: 19,
        color: ink, marginBottom: 8, lineHeight: 1.7,
      }}>
        Jiayu Chen<sup>1</sup>
        {' '}Junbei Tang<sup>2</sup>
        {' '}Wenbiao Zhao<sup>3</sup>
        {' '}Maoliang Li<sup>1</sup>
        <br/>
        Jiayi Luo<sup>4,5</sup>
        {' '}Zihao Zheng<sup>1</sup>
        {' '}Jiawei Yang<sup>1</sup>
        {' '}Guojie Luo<sup>1</sup>
        {' '}Xiang Chen<sup>1</sup>
      </div>
      <div style={{
        textAlign: 'center', fontFamily: serif, fontSize: 16,
        color: muted, marginBottom: 36, lineHeight: 1.6,
      }}>
        <sup>1</sup>Peking University{' '}
        <sup>2</sup>South China University of Technology{' '}
        <sup>3</sup>Xinjiang University
        <br/>
        <sup>4</sup>Beihang University{' '}
        <sup>5</sup>Zhongguancun Academy
      </div>

      {/* Buttons */}
      <div style={{
        display: 'flex', justifyContent: 'center', gap: 12, marginBottom: 56, flexWrap: 'wrap',
      }}>
        <Btn icon={arxivIcon} label="arXiv" href="https://arxiv.org/" bg="#B31B1B" />
        <Btn icon={githubIcon} label="Code" href="https://github.com/IF-LAB-PKU/Pyramid-Forcing" />
      </div>

      {/* Hero figure — main qualitative comparison (paper Fig. 6) */}
      <div style={{
        border: `1px solid ${rule}`, borderRadius: 8, overflow: 'hidden',
        marginBottom: 64, background: bg,
      }}>
        <img
          src={media.heroFig}
          alt="Frame-extraction comparison across methods at long-horizon timesteps."
          style={{ display: 'block', width: '100%', height: 'auto' }}
        />
      </div>

      {/* Abstract */}
      <SectionTitle>Abstract</SectionTitle>
      <p style={{
        fontFamily: serif, fontSize: 19, lineHeight: 1.65, color: ink,
        margin: '0 0 56px', textWrap: 'pretty',
      }}>
        Autoregressive video generation enables streaming and open-ended long video synthesis,
        but still suffers from long-term degradation caused by accumulated errors. Existing
        KVCache strategies usually apply unified historical-frame retention, implicitly
        assuming homogeneous historical dependencies across attention heads. We revisit
        historical-frame attention and reveal three distinct head types: Anchor Heads require
        broad long-range context, Wave Heads exhibit periodic temporal dependencies, and Veil
        Heads focus on initial and adjacent frames. Based on this finding, we propose{' '}
        <strong>Pyramid Forcing</strong>, a head-aware pyramidal KVCache framework that
        identifies head types offline, assigns behavior-specific cache policies, and supports
        heterogeneous cache lengths via efficient ragged-cache attention. Experiments on Self
        Forcing and Causal Forcing show that Pyramid Forcing consistently improves long-horizon
        generation quality on VBench-Long, increasing the 60-second Self Forcing score from
        77.87 to 81.21 while enhancing motion dynamics, visual fidelity, and semantic
        consistency.
      </p>

      {/* Analysis */}
      <SectionTitle>Head-Level Analysis</SectionTitle>
      <div style={{
        border: `1px solid ${rule}`, borderRadius: 6,
        background: bg, marginBottom: 16, overflow: 'hidden',
      }}>
        <img
          src={media.analysis}
          alt="Head-type attention patterns: anchor, wave, and veil."
          style={{ display: 'block', width: '100%', height: 'auto' }}
        />
      </div>
      <p style={{
        fontFamily: serif, fontSize: 18, lineHeight: 1.65,
        color: muted, fontStyle: 'italic', textAlign: 'center', marginBottom: 56,
      }}>
        Figure 2. Head-type attention patterns. Anchor heads spread attention broadly across the rollout, wave heads exhibit periodic structure along the time axis, and veil heads concentrate on the first and most recent frames — the three regimes the pyramid budget is built around.
      </p>

      {/* Method */}
      <SectionTitle>Method Overview</SectionTitle>
      <div style={{
        border: `1px solid ${rule}`, borderRadius: 6,
        background: bg, marginBottom: 16, overflow: 'hidden',
      }}>
        <img
          src={media.pipeline}
          alt="Pipeline: (a) head profiling → (b) pyramid budget assignment → (c) streaming inference with bounded KV"
          style={{ display: 'block', width: '100%', height: 'auto' }}
        />
      </div>
      <p style={{
        fontFamily: serif, fontSize: 18, lineHeight: 1.65,
        color: muted, fontStyle: 'italic', textAlign: 'center', marginBottom: 56,
      }}>
        Figure 3. Three-stage pipeline. Offline head profiling labels each head as anchor, wave, or veil; pyramid budget assignment derives a per-head KV size from the role; streaming inference applies the ragged-cache kernel under the bounded budget.
      </p>

      {/* Qualitative grid */}
      <SectionTitle>Qualitative Results</SectionTitle>
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        gap: 12, marginBottom: 56,
      }}>
        {media.movieGen.map((c, i) => (
          <div key={i} style={{
            border: `1px solid ${rule}`, borderRadius: 6,
            overflow: 'hidden', background: bg,
          }}>
            <video
              src={c.src}
              autoPlay loop muted playsInline controls preload="metadata"
              style={{ display: 'block', width: '100%', aspectRatio: '832 / 480' }}
            />
            <div
              title={c.prompt}
              style={{
                padding: '10px 14px', borderTop: `1px solid ${rule}`,
                fontFamily: serif, fontSize: 13, fontStyle: 'italic', color: ink,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}
            >
              “{c.prompt}”
            </div>
          </div>
        ))}
      </div>

      {/* Baseline comparison — 30 s rollouts, 2 rows × 4 columns. */}
      <SectionTitle>Comparison with Baselines</SectionTitle>
      <div style={{
        fontFamily: serif, fontSize: 22, fontWeight: 600,
        color: ink, marginTop: -4, marginBottom: 16,
      }}>30-second video</div>
      {media.compare30s.map((row, ri) => {
        const isLast = ri === media.compare30s.length - 1;
        return (
          <React.Fragment key={`c30-${ri}`}>
            <div style={{
              display: 'flex', alignItems: 'baseline', gap: 8,
              fontFamily: sans, fontSize: 12, color: muted, letterSpacing: 0.5,
              marginBottom: 14,
            }}>
              <span style={{ flexShrink: 0 }}>PROMPT</span>
              <span
                title={row.prompt}
                style={{
                  color: ink, fontFamily: serif, fontSize: 15, fontStyle: 'italic',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  flex: 1, minWidth: 0,
                }}
              >“{row.prompt}”</span>
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
              gap: 12, marginBottom: isLast ? 56 : 28,
            }}>
              {row.cells.map((c, ci) => (
                <div key={ci} style={{
                  border: `1px solid ${rule}`, borderRadius: 6,
                  overflow: 'hidden', background: bg,
                }}>
                  <video
                    src={c.src}
                    autoPlay loop muted playsInline controls preload="metadata"
                    style={{ display: 'block', width: '100%', aspectRatio: '832 / 480' }}
                  />
                  <div style={{
                    padding: '10px 14px', borderTop: `1px solid ${rule}`,
                    fontFamily: sans, fontSize: 13,
                    color: c.highlight ? navy : muted,
                    fontWeight: c.highlight ? 600 : 400,
                    letterSpacing: 0.2,
                  }}>
                    {c.highlight && <span style={{ marginRight: 4 }}>★</span>}
                    {c.label}
                  </div>
                </div>
              ))}
            </div>
          </React.Fragment>
        );
      })}

      {/* Baseline comparison — 60 s rollouts, same 2 × 4 layout (no second h3, just a timing label). */}
      <div style={{
        fontFamily: serif, fontSize: 22, fontWeight: 600,
        color: ink, marginBottom: 16,
      }}>60-second video</div>
      {media.compare60s.map((row, ri) => {
        const isLast = ri === media.compare60s.length - 1;
        return (
          <React.Fragment key={`c60-${ri}`}>
            <div style={{
              display: 'flex', alignItems: 'baseline', gap: 8,
              fontFamily: sans, fontSize: 12, color: muted, letterSpacing: 0.5,
              marginBottom: 14,
            }}>
              <span style={{ flexShrink: 0 }}>PROMPT</span>
              <span
                title={row.prompt}
                style={{
                  color: ink, fontFamily: serif, fontSize: 15, fontStyle: 'italic',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  flex: 1, minWidth: 0,
                }}
              >“{row.prompt}”</span>
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
              gap: 12, marginBottom: isLast ? 56 : 28,
            }}>
              {row.cells.map((c, ci) => (
                <div key={ci} style={{
                  border: `1px solid ${rule}`, borderRadius: 6,
                  overflow: 'hidden', background: bg,
                }}>
                  <video
                    src={c.src}
                    autoPlay loop muted playsInline controls preload="metadata"
                    style={{ display: 'block', width: '100%', aspectRatio: '832 / 480' }}
                  />
                  <div style={{
                    padding: '10px 14px', borderTop: `1px solid ${rule}`,
                    fontFamily: sans, fontSize: 13,
                    color: c.highlight ? navy : muted,
                    fontWeight: c.highlight ? 600 : 400,
                    letterSpacing: 0.2,
                  }}>
                    {c.highlight && <span style={{ marginRight: 4 }}>★</span>}
                    {c.label}
                  </div>
                </div>
              ))}
            </div>
          </React.Fragment>
        );
      })}

      {/* Ablation — same row/column form as the comparison sections above. */}
      <SectionTitle>Ablation</SectionTitle>
      {media.ablation.map((row, ri) => {
        const isLast = ri === media.ablation.length - 1;
        return (
          <React.Fragment key={`abl-${ri}`}>
            <div style={{
              display: 'flex', alignItems: 'baseline', gap: 8,
              fontFamily: sans, fontSize: 12, color: muted, letterSpacing: 0.5,
              marginBottom: 14,
            }}>
              <span style={{ flexShrink: 0 }}>PROMPT</span>
              <span
                title={row.prompt}
                style={{
                  color: ink, fontFamily: serif, fontSize: 15, fontStyle: 'italic',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  flex: 1, minWidth: 0,
                }}
              >“{row.prompt}”</span>
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
              gap: 12, marginBottom: isLast ? 56 : 28,
            }}>
              {row.cells.map((c, ci) => (
                <div key={ci} style={{
                  border: `1px solid ${rule}`, borderRadius: 6,
                  overflow: 'hidden', background: bg,
                }}>
                  <video
                    src={c.src}
                    autoPlay loop muted playsInline controls preload="metadata"
                    style={{ display: 'block', width: '100%', aspectRatio: '832 / 480' }}
                  />
                  <div style={{
                    padding: '10px 14px', borderTop: `1px solid ${rule}`,
                    fontFamily: sans, fontSize: 13,
                    color: c.highlight ? navy : muted,
                    fontWeight: c.highlight ? 600 : 400,
                    letterSpacing: 0.2,
                  }}>
                    {c.highlight && <span style={{ marginRight: 4 }}>★</span>}
                    {c.label}
                  </div>
                </div>
              ))}
            </div>
          </React.Fragment>
        );
      })}

      {/* Footer / cite */}
      <div style={{
        borderTop: `1px solid ${rule}`, paddingTop: 32,
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32,
      }}>
        <div>
          <div style={{
            fontFamily: sans, fontSize: 11, letterSpacing: 1.5,
            color: navy, textTransform: 'uppercase', fontWeight: 600, marginBottom: 8,
          }}>BibTeX</div>
          <pre style={{
            fontFamily: mono, fontSize: 11, lineHeight: 1.5,
            background: tint, border: `1px solid ${rule}`,
            padding: '14px 16px', color: ink, margin: 0,
            whiteSpace: 'pre-wrap',
          }}>{`Coming soon.`}</pre>
        </div>
        <div>
          <div style={{
            fontFamily: sans, fontSize: 11, letterSpacing: 1.5,
            color: navy, textTransform: 'uppercase', fontWeight: 600, marginBottom: 8,
          }}>Acknowledgements</div>
          <p style={{
            fontFamily: serif, fontSize: 15, lineHeight: 1.6, color: muted, margin: 0,
          }}>
            Page template adapted from the Nerfies project page. Thanks to the open-source
            long-video generation community for the models and benchmarks this work builds on.
          </p>
        </div>
      </div>
    </div>
  );
};

const SectionTitle = ({ children, subtitle }) => (
  <div style={{ marginBottom: 18 }}>
    <h3 style={{
      fontFamily: `'Source Serif 4', Georgia, serif`,
      fontWeight: 600, fontSize: 32, margin: 0,
      letterSpacing: -0.3,
    }}>{children}</h3>
    {subtitle && (
      <div style={{
        fontFamily: `'Source Serif 4', Georgia, serif`,
        fontSize: 16, fontStyle: 'italic',
        color: 'oklch(0.45 0.01 250)',
        marginTop: 6,
      }}>{subtitle}</div>
    )}
    <div style={{
      width: 36, height: 2, background: 'oklch(0.42 0.09 250)',
      marginTop: 10,
    }} />
  </div>
);

window.PyramidForcingApp = PyramidForcingApp;
