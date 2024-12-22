from filter.contrast_enhancement_technique.gamma_correction import gamma_correction
from filter.contrast_enhancement_technique.histogram_equalization import histogram_equalization
from filter.contrast_enhancement_technique.luminance_adaption import luminance_adaption

def contrast_enhancement(image, techniques):
    for technique in techniques:
        if technique == 'g':
            image = gamma_correction(image)
        if technique == 'h':
            image = histogram_equalization(image)
        if technique == 'l':
            image = luminance_adaption(image)

    return image