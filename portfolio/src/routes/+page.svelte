<script>
    import Intro from '$lib/components/Intro.svelte';
	import Activities from '$lib/components/Activities.svelte';
	import Resources from '$lib/components/Resources.svelte';
	import Divider from '$lib/components/Divider.svelte';
	import About from '$lib/components/About.svelte';
    import { dataStore } from '../stores/dataStore';

    // Create a readable specialization string for the meta description
    let personal = $dataStore.PERSONAL;
    let specializations = $dataStore.SKILLS.specializations.join(', ');
</script>

<svelte:head>
    <title>{personal.name} - {personal.role}</title>
    <meta name="description" content="{personal.name} - {personal.role} specializing in {specializations}. View my projects and contact info.">
</svelte:head>

<Intro />
<Divider />
<Resources />
<Divider />
<Activities />
<Divider />
<About />
<Divider />


