# skeleton-depth

**Pure-Rust upper-body skeleton tracking from a depth silhouette** — head,
shoulders and hands, with **zero dependencies** (no OpenCV, no OpenNI, no ML).

🇬🇧 [English](#english) · 🇫🇷 [Français](#français)

<a id="english"></a>

## 🇬🇧 English

Feed a depth frame (millimeters) and get back the **head, shoulders and hands**
of the person standing closest to the camera. Cheap enough to run every frame on
one core, and the head is anchored to the body — not "the nearest blob" — so a
nearer object off to the side isn't mistaken for the head.

### Input

The crate is **I/O-free**: it never opens a sensor. You hand it a raw depth
buffer and it does the rest.

- **Format**: `&[u16]`, row-major, one value per pixel, in **millimeters**,
  `0` = no data. Length must be `width * height`.
- **Kinect v1** — via [`libfreenect`](https://github.com/OpenKinect/libfreenect):
  depth is 640×480 `u16` millimeters. Pass the buffer straight through.
- **Kinect v2** — via [`libfreenect2`](https://github.com/OpenKinect/libfreenect2):
  depth is 512×424 `f32` millimeters; cast each sample to `u16`
  (`v as u16`, `0.0` stays `0`) and pass it in.
- **Anything else** — RealSense, other ToF/structured-light sensors, or a
  recorded frame all work: it's just an array of millimeters.
- **No depth?** A webcam with a smooth background can produce a silhouette by
  background subtraction and call `Tracker::track_mask` instead — the same
  joint rules then work in 2D.

```rust
use skeleton_depth::{Config, Tracker};

let (w, h) = (512, 424);                 // Kinect v2 depth resolution
let mut tracker = Tracker::new(w, h, Config::default());

// each frame: depth in u16 millimeters, 0 = no data
let skel = tracker.track(&depth);
```

### Output

A `Skeleton` for the frame. Every joint is an `Option` — `None` when that part
wasn't found this frame:

| Field | Meaning |
|---|---|
| `head` | top of the head |
| `left_shoulder`, `right_shoulder` | shoulders |
| `left_hand`, `right_hand` | hands (outermost silhouette points) |
| `left_elbow`, `right_elbow` | elbows — *roadmap, always `None` for now* |
| `center` | silhouette centroid (torso anchor) |

Each `Joint` is a full-resolution pixel plus depth: `{ x, y, z_mm }`.
`Joint::to_metric(&Intrinsics)` deprojects it to camera-space millimeters
`[X, Y, Z]` through pinhole intrinsics.

```rust
if let Some(head) = skel.head {
    println!("head @ ({}, {}) — {} mm deep", head.x, head.y, head.z_mm);
}
if let (Some(l), Some(r)) = (skel.left_hand, skel.right_hand) {
    // e.g. which body is the player? the one whose hands are on the controls.
}
```

### How it works

1. **Segment**: keep the near depth slab `[closest, closest + slab_mm]` as
   foreground → a binary silhouette. *(The only depth-dependent stage.)*
2. **Isolate**: keep the largest connected component (drops bystanders/speckle).
3. **Extremities**: relative to the body's centre column, the **head** is the
   highest silhouette pixel and the **hands** are the outermost pixels in the
   left/right quadrants; **shoulders** drop from the head column to the edge.
4. **Smooth**: a short per-joint temporal median rejects single-frame outliers.

### Status / roadmap

Working: **head, shoulders, hands, body centre**, with temporal smoothing.
Not yet ported: **elbows** (arm-curve angle tracking); **adaptive slab**;
optional **geodesic head selection** (connectivity to the body mass). The tuning
constants mirror a frontal camera and will want retuning for very different
placements.

<a id="français"></a>

## 🇫🇷 Français

Donne-lui une frame de profondeur (en millimètres) et récupère la **tête, les
épaules et les mains** de la personne la plus proche de la caméra. Assez léger
pour tourner à chaque frame sur un seul cœur, et la tête est **ancrée au corps**
— pas « le blob le plus proche » — donc un objet plus proche mais sur le côté
n'est pas pris pour la tête.

### Entrée

Le crate est **sans I/O** : il n'ouvre aucun capteur. Tu lui passes un buffer de
profondeur brut, il fait le reste.

- **Format** : `&[u16]`, ligne par ligne, une valeur par pixel, en
  **millimètres**, `0` = pas de donnée. Taille = `largeur * hauteur`.
- **Kinect v1** — via [`libfreenect`](https://github.com/OpenKinect/libfreenect) :
  profondeur 640×480 en `u16` mm. Le buffer passe tel quel.
- **Kinect v2** — via [`libfreenect2`](https://github.com/OpenKinect/libfreenect2) :
  profondeur 512×424 en `f32` mm ; caste chaque échantillon en `u16`
  (`v as u16`, `0.0` reste `0`) et passe-le.
- **Autre chose** — RealSense, autres capteurs ToF/lumière structurée, ou une
  frame enregistrée : c'est juste un tableau de millimètres.
- **Pas de profondeur ?** Une webcam à fond lisse peut produire une silhouette
  par soustraction de fond et appeler `Tracker::track_mask` à la place — les
  mêmes règles de joints fonctionnent alors en 2D.

### Sortie

Un `Skeleton` pour la frame. Chaque joint est un `Option` — `None` si la partie
n'a pas été trouvée cette frame :

| Champ | Signification |
|---|---|
| `head` | sommet de la tête |
| `left_shoulder`, `right_shoulder` | épaules |
| `left_hand`, `right_hand` | mains (points extrêmes de la silhouette) |
| `left_elbow`, `right_elbow` | coudes — *roadmap, `None` pour l'instant* |
| `center` | centroïde de la silhouette (ancre torse) |

Chaque `Joint` est un pixel pleine résolution + profondeur : `{ x, y, z_mm }`.
`Joint::to_metric(&Intrinsics)` le déprojette en millimètres caméra `[X, Y, Z]`
via des intrinsèques sténopé.

### Fonctionnement

1. **Segmenter** : garder la tranche proche `[proche, proche + slab_mm]` comme
   avant-plan → silhouette binaire. *(Seule étape dépendante de la profondeur.)*
2. **Isoler** : garder la plus grande composante connexe (vire spectateurs/bruit).
3. **Extrémités** : par rapport à la colonne centrale du corps, la **tête** est
   le pixel le plus haut et les **mains** les pixels les plus externes des
   quadrants gauche/droite ; les **épaules** descendent de la tête jusqu'au bord.
4. **Lisser** : une médiane temporelle courte par joint rejette les aberrations.

### État / roadmap

Fonctionne : **tête, épaules, mains, centre du corps**, avec lissage temporel.
Pas encore porté : **coudes** (suivi d'angle du bras) ; **slab adaptatif** ;
option **tête géodésique** (connexité à la masse corps). Les constantes de tuning
reflètent une caméra frontale et demanderont un réglage pour d'autres placements.

---

## 🙏 Acknowledgements / Remerciements

🇬🇧 This crate is a **from-scratch Rust reimplementation** of the upper-body
skeleton-detection method by **[derzu](https://github.com/derzu/BodySkeletonTracker/commits?author=derzu)**
in his original C++ program **[BodySkeletonTracker](https://github.com/derzu/BodySkeletonTracker)**
(MIT, © 2018 Derzu). Huge thanks to him for designing and sharing such a simple,
elegant, dependency-light approach — no original source code was copied, only the
method was reimplemented. The underlying idea traces to Andreas Baak's *"A
Data-Driven Approach for Real-Time Full Body Pose Reconstruction from a Depth
Camera"*.

🇫🇷 Ce crate est une **réimplémentation Rust from-scratch** de la méthode de
détection de squelette haut-du-corps de **[derzu](https://github.com/derzu/BodySkeletonTracker/commits?author=derzu)**,
tirée de son programme C++ d'origine **[BodySkeletonTracker](https://github.com/derzu/BodySkeletonTracker)**
(MIT, © 2018 Derzu). Un grand merci à lui pour avoir conçu et partagé une
approche aussi simple, élégante et légère en dépendances — aucun code source
d'origine n'a été copié, seule la méthode a été réimplémentée. L'idée de fond
vient du papier d'Andreas Baak, *« A Data-Driven Approach for Real-Time Full Body
Pose Reconstruction from a Depth Camera »*.

## License

MIT — see [`LICENSE-MIT`](LICENSE-MIT).
