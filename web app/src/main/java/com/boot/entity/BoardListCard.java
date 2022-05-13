package com.boot.entity;

import java.io.Serializable;
import java.util.Date;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

import com.fasterxml.jackson.annotation.JsonFormat;

import lombok.Data;

/**
 * board list card
 */
@Entity
@Table(name = "t_board_list_card")
@Data
public class BoardListCard implements Serializable{
	private static final long serialVersionUID = 7515656429107760389L;
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Integer id;
	private String name;
	
	@Column(name="list_id")
	private Integer listId;//
	
	private String description;
	
	@Column(name="due_time")
	private Date dueTime;
	
	@Column(name="start_time")
	@JsonFormat(pattern="yyyy-MM-dd HH:mm",timezone="GMT+8")
	private Date startTime;
	
	@Column(name="end_time")
	@JsonFormat(pattern="yyyy-MM-dd HH:mm",timezone="GMT+8")
	private Date endTime;
	
	private Date attachments;
	
	@Column(name="attachments_status")
	private Date attachmentsStatus;
	
	@Column(name="create_time",updatable=false)
	@JsonFormat(pattern="yyyy-MM-dd HH:mm",timezone="GMT+8")
	private Date createTime = new Date();//创建时间
}
