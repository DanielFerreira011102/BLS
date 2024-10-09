<script>
    import DATA from '../data';
    import Intro from '$lib/components/Intro.svelte';

    // Create a readable specialization string for the meta description
    let specializations = DATA.SKILLS.specializations.join(', ');
</script>

<svelte:head>
    <title>{DATA.PERSONAL.name} - {DATA.PERSONAL.role}</title>
    <meta name="description" content="{DATA.PERSONAL.name} - {DATA.PERSONAL.role} specializing in {specializations}. View my projects and contact info.">
</svelte:head>

<Intro personal={DATA.PERSONAL} />


