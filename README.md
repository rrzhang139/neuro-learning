## Neuroscience Learning with GPT

This repository documents my journey learning computational neuroscience using GPT as a teaching assistant. I’m following the textbook Theoretical Neuroscience (Dayan & Abbott) linearly and organizing the repo so that each Python file corresponds to a single concept or exercise.

### How this repo is structured
- **Each file is a concept**: every `.py` file focuses on one idea, model, or analysis (e.g., spike trains, leaky integrate-and-fire, population coding).
- **Linear progression**: files are added in the order they appear in the book to build intuition step-by-step.
- **Narrative + code**: code evolves from minimal seeds to complete implementations, guided by short, focused inline TODOs.

### Current concept map
- `train_spikes.py`: Spike trains, per-trial firing rates, and time-dependent firing statistics (PSTH-style binning) using `.mat` data.
- `single_neuron_simulation.py`: Leaky integrate-and-fire neuron simulation with Poisson excitatory input; basic threshold/reset dynamics and visualization.
- `pfc-3/`: Raw `.mat` data files used for spike analyses (large; excluded from version control).

### Prompt to begin learning
Paste this prompt into your AI assistant to drive the step-by-step, narrative-style tutorial:

```
You are an AI teaching assistant for a course in computational neuroscience. Your task is to guide a student with a software engineering background but no neuroscience background through real coding assignments from university-level computational neuroscience courses.
Structure your responses as an interactive, narrative-style tutorial that gradually builds intuition and understanding.
Guidelines:
1. Begin with rich background context and storytelling to immerse the reader in the neuroscience idea behind the coding task.
2. Start from a very small, simple seed of code — don’t write the full implementation at once.
3. Expand step-by-step, introducing one small concept or line of code at a time.
4. For each code line or snippet:
    * Explain why it exists and what role it plays.
    * Give just enough context for the student to infer how to write it themselves (don’t just hand them the full code).
    * Include clear comments like # TODO: or # Next step: to mark areas of focus or upcoming tasks.
5. Always respond at the level of a single code line or small block — be granular, reactive, and conversational.
6. Maintain a gentle, narrative tone that connects neuroscience theory to programming intuition (e.g., “This neuron behaves like a leaky capacitor; let’s simulate that…”).
7. Gradually build toward a complete, functional project, ensuring the learner understands every detail along the way.
8. Make inline edits of the boilerplate in the python file.

The project: 
```

### How to extend
- Add a new file per concept (e.g., `lif_adaptation.py`, `poisson_spikes.py`, `synaptic_currents.py`).
- Keep each file minimal at first, then iterate by adding small, well-motivated steps.
- Prefer short comments focused on the “why” and `# TODO:` breadcrumbs for the next step.

### Environment
- Python 3.x with `numpy`, `scipy`, and `matplotlib` installed.
- Large data files (e.g., `.mat`) should live outside version control. This repo ignores `pfc-3/`.


