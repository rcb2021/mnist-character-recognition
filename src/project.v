/*
 * FEASIBILITY PROTOTYPE - greyscale-input variant of tt_um_kenchangh_mnist.
 *
 * Derived from src/project.v of github.com/kenchangh/mnist-character-recognition
 * (commit 33c215e, Copyright (c) 2026 kenchangh, Apache-2.0), fixed to the
 * 64 -> 32 -> 10 variant. Changes from the original:
 *   - pixel is PIXW bits wide (1, 2 or 4) instead of 1 bit
 *   - pixel arrives in parallel on ui_in[4 +: PIXW] (original: serial on ui_in[0])
 *   - input store is 64*PIXW bits (original: 64 bits)
 *   - layer 1 adds / subtracts the pixel value (original: adds / subtracts 1)
 *   - accumulators are AW bits (8 for PIXW=1, 9 for PIXW=2, 11 for PIXW=4)
 * Unchanged: state machine, two 16-lane passes over layer 1, layer 2,
 * sequential argmax, 7-segment decode, SHIFT_EN / START / DONE / BUSY pins.
 * With PIXW=1, AW=8 it is functionally the original design.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_proto_mnist_grey #(
    parameter PIXW = 4,   // bits per pixel
    parameter AW   = 11   // accumulator width (signed)
) (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  localparam HIDDEN = 32;
  localparam LANES  = 16;
  localparam [5:0] L2_LAST = HIDDEN - 1;
  localparam IW = 64 * PIXW;

  localparam [2:0] S_IDLE   = 3'd0,
                   S_LOAD   = 3'd1,
                   S_L1     = 3'd2,
                   S_BIN    = 3'd3,
                   S_L2     = 3'd4,
                   S_ARGMAX = 3'd5,
                   S_DONE   = 3'd6;

  wire [PIXW-1:0] data_in  = ui_in[4 +: PIXW];
  wire            shift_en = ui_in[1];
  wire            start    = ui_in[2];

  reg  [2:0]           state;
  reg  [5:0]           cnt;
  reg  [IW-1:0]        inp;         // pixel i at bits [i*PIXW +: PIXW] once loaded
  reg  signed [AW-1:0] acc [0:15];
  reg  [31:0]          h;
  reg                  pass;
  reg  signed [AW-1:0] best_val;
  reg  [3:0]           best_idx;
  reg  [3:0]           digit;
  reg                  done_r;
  reg                  have_result;

  wire [31:0]        l1_pos, l1_neg;
  wire [32*AW-1:0]   b1_flat;
  wire [9:0]         l2_pos, l2_neg;
  wire [10*AW-1:0]   b2_flat;

  mnist_weights u_weights (
      .idx(cnt), .hidx(cnt[4:0]),
      .l1_pos(l1_pos), .l1_neg(l1_neg), .b1_flat(b1_flat),
      .l2_pos(l2_pos), .l2_neg(l2_neg), .b2_flat(b2_flat));

  wire [PIXW-1:0]      pix   = inp[PIXW-1:0];
  wire signed [AW-1:0] pix_s = $signed({{(AW-PIXW){1'b0}}, pix});
  wire signed [AW-1:0] one_s = $signed({{(AW-1){1'b0}}, 1'b1});
  wire signed [AW-1:0] min_s = $signed({1'b1, {(AW-1){1'b0}}});

  wire [15:0] w1p_sel = pass ? l1_pos[31:16] : l1_pos[15:0];
  wire [15:0] w1n_sel = pass ? l1_neg[31:16] : l1_neg[15:0];

  integer i;

  always @(posedge clk) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      cnt         <= 6'd0;
      inp         <= {IW{1'b0}};
      h           <= 32'd0;
      best_val    <= {AW{1'b0}};
      best_idx    <= 4'd0;
      digit       <= 4'd0;
      done_r      <= 1'b0;
      have_result <= 1'b0;
      pass        <= 1'b0;
      for (i = 0; i < 16; i = i + 1) acc[i] <= {AW{1'b0}};
    end else if (start) begin
      state  <= S_LOAD;
      cnt    <= 6'd0;
      pass   <= 1'b0;
      done_r <= 1'b0;
    end else begin
      case (state)
        S_LOAD: begin
          if (shift_en) begin
            inp <= {data_in, inp[IW-1:PIXW]};
            cnt <= cnt + 6'd1;
            if (cnt == 6'd63) begin
              state <= S_L1;
              cnt   <= 6'd0;
              pass  <= 1'b0;
              for (i = 0; i < LANES; i = i + 1)
                acc[i] <= $signed(b1_flat[i*AW +: AW]);
            end
          end
        end

        S_L1: begin
          inp <= {inp[PIXW-1:0], inp[IW-1:PIXW]};  // rotate by one pixel
          for (i = 0; i < LANES; i = i + 1) begin
            if (w1p_sel[i])      acc[i] <= acc[i] + pix_s;
            else if (w1n_sel[i]) acc[i] <= acc[i] - pix_s;
          end
          cnt <= cnt + 6'd1;
          if (cnt == 6'd63) begin
            cnt   <= 6'd0;
            state <= S_BIN;
          end
        end

        S_BIN: begin
          for (i = 0; i < LANES; i = i + 1) begin
            if (pass == 1'b0) h[i] <= ~acc[i][AW-1];
            else              h[16 + i] <= ~acc[i][AW-1];
          end
          cnt <= 6'd0;
          if (pass == 1'b0) begin
            pass  <= 1'b1;
            state <= S_L1;
            for (i = 0; i < LANES; i = i + 1)
              acc[i] <= $signed(b1_flat[(i + 16)*AW +: AW]);
          end else begin
            state <= S_L2;
            for (i = 0; i < 10; i = i + 1)
              acc[i] <= $signed(b2_flat[i*AW +: AW]);
          end
        end

        S_L2: begin
          for (i = 0; i < 10; i = i + 1) begin
            if (h[cnt[4:0]] && l2_pos[i])      acc[i] <= acc[i] + one_s;
            else if (h[cnt[4:0]] && l2_neg[i]) acc[i] <= acc[i] - one_s;
          end
          cnt <= cnt + 6'd1;
          if (cnt == L2_LAST) begin
            cnt      <= 6'd0;
            state    <= S_ARGMAX;
            best_val <= min_s;
            best_idx <= 4'd0;
          end
        end

        S_ARGMAX: begin
          if (acc[cnt[3:0]] > best_val) begin
            best_val <= acc[cnt[3:0]];
            best_idx <= cnt[3:0];
          end
          cnt <= cnt + 6'd1;
          if (cnt == 6'd9) begin
            digit       <= (acc[9] > best_val) ? 4'd9 : best_idx;
            done_r      <= 1'b1;
            have_result <= 1'b1;
            state       <= S_DONE;
          end
        end

        S_DONE: ;

        default: state <= S_IDLE;
      endcase
    end
  end

  reg [6:0] seg;
  always @(*) begin
    case (digit)
      4'd0: seg = 7'h3F;
      4'd1: seg = 7'h06;
      4'd2: seg = 7'h5B;
      4'd3: seg = 7'h4F;
      4'd4: seg = 7'h66;
      4'd5: seg = 7'h6D;
      4'd6: seg = 7'h7D;
      4'd7: seg = 7'h07;
      4'd8: seg = 7'h7F;
      4'd9: seg = 7'h6F;
      default: seg = 7'h00;
    endcase
  end

  wire busy = (state == S_L1) || (state == S_BIN) ||
              (state == S_L2) || (state == S_ARGMAX);

  assign uo_out  = {done_r, have_result ? seg : 7'h00};
  assign uio_out = {2'b00, busy, done_r, digit};
  assign uio_oe  = 8'h3F;

  wire _unused = &{ena, ui_in[3], ui_in[0], uio_in, 1'b0};

endmodule
