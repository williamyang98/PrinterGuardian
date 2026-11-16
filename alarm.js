class Alarm {
    constructor() {
        this.context = new AudioContext();
        this.is_armed = false;

        this.sample_rate = this.context.sampleRate;
        const total_channels = 1;
        const total_duration = 1.0;
        const total_samples = Math.floor(this.sample_rate*total_duration);

        this.buffer = this.context.createBuffer(total_channels, total_samples, this.sample_rate);
        const data = this.buffer.getChannelData(0);

        // generate on and off square wave
        const total_on_duration = 0.3;
        const total_on_samples = Math.floor(this.sample_rate*total_on_duration);
        const frequency_hz = 1000;
        const amplitude = 0.8;
        for (let i = 0; i < total_on_samples; i++) {
            const time = i/this.sample_rate;
            const x = 2*Math.PI*frequency_hz*time;
            let y = Math.sin(x);
            y = Math.sign(y); // turn into square wave
            y = y*amplitude;
            data[i] = y;
        }
        for (let i = total_on_samples; i < total_samples; i++) {
            data[i] = 0.0;
        }
        this.source = null;
    }

    // chrome does not an audio context to resume without user interaction
    arm() {
        this.context.resume();
        this.is_armed = true;
    }

    play() {
        this.stop();
        this.source = this.context.createBufferSource();
        this.source.buffer = this.buffer;
        this.source.loop = true;
        this.source.connect(this.context.destination);
        this.source.start(0);
    }

    stop() {
        if (this.source === null) return;
        this.source.stop();
        this.source.disconnect();
        this.source = null;
    }
}